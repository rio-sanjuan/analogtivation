"""Cross-framework consistency tests."""

import pytest
import numpy as np
from unittest.mock import patch
from time import struct_time


class TestCrossFrameworkConsistency:
    """Test consistency across different framework implementations."""
    
    def get_test_inputs(self):
        """Get standard test inputs for consistency checks."""
        return {
            "small": np.array([-2, -1, 0, 1, 2], dtype=np.float32),
            "random": np.random.randn(100).astype(np.float32),
            "edge": np.array([0, 1e-7, -1e-7, 10, -10], dtype=np.float32),
        }
    
    @pytest.mark.parametrize("activation_name", [
        "wave_activation",
        "fourier_activation",
        "quantum_activation",
        "lorenz_activation",
        "logistic_map_activation",
        "mandelbrot_activation",
    ])
    def test_numpy_tensorflow_consistency(self, activation_name, frameworks_available, skip_if_no_tensorflow):
        """Test consistency between NumPy and TensorFlow implementations."""
        import tensorflow as tf
        from analogtivation import core
        from analogtivation import tensorflow as tf_impl
        
        test_inputs = self.get_test_inputs()
        
        # Get activation functions
        numpy_func = getattr(core, activation_name)
        tf_layer = getattr(tf_impl, activation_name.replace("_activation", "").title() + "Activation")
        
        for input_name, x in test_inputs.items():
            # NumPy result
            numpy_result = numpy_func(x)
            
            # TensorFlow result
            layer = tf_layer()
            tf_input = tf.constant(x)
            tf_result = layer(tf_input).numpy()
            
            # Compare results
            np.testing.assert_allclose(
                numpy_result, tf_result,
                rtol=1e-5, atol=1e-5,
                err_msg=f"Mismatch for {activation_name} with {input_name} input"
            )
    
    @pytest.mark.parametrize("activation_name", [
        "wave_activation",
        "fourier_activation",
        "quantum_activation",
        "lorenz_activation",
        "logistic_map_activation",
        "mandelbrot_activation",
    ])
    def test_numpy_torch_consistency(self, activation_name, frameworks_available, skip_if_no_torch):
        """Test consistency between NumPy and PyTorch implementations."""
        import torch
        from analogtivation import core
        from analogtivation.torch import functional as torch_func
        
        test_inputs = self.get_test_inputs()
        
        # Get activation functions
        numpy_func = getattr(core, activation_name)
        torch_activation = getattr(torch_func, activation_name)
        
        for input_name, x in test_inputs.items():
            # NumPy result
            numpy_result = numpy_func(x)
            
            # PyTorch result
            torch_input = torch.tensor(x)
            
            # Handle functions with different signatures
            if activation_name == "fourier_activation":
                weights = torch.ones(3)
                torch_result = torch_activation(torch_input, weights, n_harmonics=3).numpy()
            elif activation_name == "quantum_activation":
                state_amplitudes = torch.randn(2) * 0.1
                torch_result = torch_activation(torch_input, state_amplitudes, n_states=2).numpy()
            else:
                torch_result = torch_activation(torch_input).numpy()
            
            # Compare results
            np.testing.assert_allclose(
                numpy_result, torch_result,
                rtol=1e-5, atol=1e-5,
                err_msg=f"Mismatch for {activation_name} with {input_name} input"
            )
    
    @pytest.mark.parametrize("activation_name", [
        "wave_activation",
        "lorenz_activation",
        "logistic_map_activation",
        "mandelbrot_activation",
    ])
    def test_numpy_jax_consistency(self, activation_name, frameworks_available, skip_if_no_jax):
        """Test consistency between NumPy and JAX implementations."""
        import jax.numpy as jnp
        from analogtivation import core
        from analogtivation.jax import functions as jax_func
        
        test_inputs = self.get_test_inputs()
        
        # Get activation functions
        numpy_func = getattr(core, activation_name)
        jax_activation = getattr(jax_func, activation_name)
        
        for input_name, x in test_inputs.items():
            # NumPy result
            numpy_result = numpy_func(x)
            
            # JAX result
            jax_input = jnp.array(x)
            jax_result = np.array(jax_activation(jax_input))
            
            # Compare results
            np.testing.assert_allclose(
                numpy_result, jax_result,
                rtol=1e-5, atol=1e-5,
                err_msg=f"Mismatch for {activation_name} with {input_name} input"
            )
    
    @patch('time.gmtime')
    def test_time_based_consistency(self, mock_gmtime, frameworks_available):
        """Test time-based activations across frameworks with fixed time."""
        # Fix time for consistency
        mock_gmtime.return_value = struct_time((2024, 6, 15, 14, 30, 0, 0, 166, 0))
        
        test_inputs = self.get_test_inputs()
        
        # Test clock activation
        if frameworks_available["tensorflow"]:
            import tensorflow as tf
            from analogtivation.core import clock_activation
            from analogtivation.tensorflow import ClockActivation
            
            for input_name, x in test_inputs.items():
                numpy_result = clock_activation(x)
                
                layer = ClockActivation()
                tf_input = tf.constant(x)
                tf_result = layer(tf_input).numpy()
                
                np.testing.assert_allclose(
                    numpy_result, tf_result,
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"Clock activation mismatch for {input_name} input"
                )
        
        if frameworks_available["torch"]:
            import torch
            from analogtivation.core import seasonal_activation
            from analogtivation.torch.functional import seasonal_activation as torch_seasonal
            
            for input_name, x in test_inputs.items():
                numpy_result = seasonal_activation(x)
                
                torch_input = torch.tensor(x)
                torch_result = torch_seasonal(torch_input).numpy()
                
                np.testing.assert_allclose(
                    numpy_result, torch_result,
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"Seasonal activation mismatch for {input_name} input"
                )
    
    def test_gradient_consistency(self, frameworks_available):
        """Test gradient computation consistency across frameworks."""
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        
        # Test wave activation gradients
        if frameworks_available["tensorflow"] and frameworks_available["torch"]:
            import tensorflow as tf
            import torch
            from analogtivation.tensorflow import WaveActivation as TFWaveActivation
            from analogtivation.torch import WaveActivation as TorchWaveActivation
            
            # TensorFlow gradient
            tf_x = tf.constant(x)
            with tf.GradientTape() as tape:
                tape.watch(tf_x)
                tf_layer = TFWaveActivation()
                tf_y = tf_layer(tf_x)
            tf_grad = tape.gradient(tf_y, tf_x).numpy()
            
            # PyTorch gradient
            torch_x = torch.tensor(x, requires_grad=True)
            torch_layer = TorchWaveActivation()
            torch_y = torch_layer(torch_x)
            torch_y.sum().backward()
            torch_grad = torch_x.grad.numpy()
            
            # Compare gradients
            np.testing.assert_allclose(
                tf_grad, torch_grad,
                rtol=1e-4, atol=1e-4,
                err_msg="Gradient mismatch between TensorFlow and PyTorch"
            )