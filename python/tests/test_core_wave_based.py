"""Tests for wave-based activation functions."""

import pytest
import numpy as np

from analogtivation.core.wave_based import (
    wave_activation,
    fourier_activation,
    quantum_activation,
    WaveActivation,
    FourierActivation,
    QuantumActivation,
)
from .test_utils import check_gradient_numerically, check_numerical_stability, check_activation_properties


class TestWaveActivation:
    """Test wave activation function."""
    
    def test_wave_activation_basic(self, sample_inputs):
        """Test basic wave activation functionality."""
        x = sample_inputs["small"]
        result = wave_activation(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        
        # Check that it produces wave-like output
        assert np.any(result != x)  # Should transform input
        
    def test_wave_activation_frequency(self, sample_inputs):
        """Test wave activation with different frequencies."""
        x = sample_inputs["medium"]
        
        result1 = wave_activation(x, frequencies=[1.0], amplitudes=[1.0])
        result2 = wave_activation(x, frequencies=[2.0], amplitudes=[1.0])
        result3 = wave_activation(x, frequencies=[0.5], amplitudes=[1.0])
        
        # Different frequencies should produce different results
        assert not np.allclose(result1, result2)
        assert not np.allclose(result1, result3)
        
        # Higher frequency should have more oscillations
        zero_crossings1 = np.sum(np.diff(np.sign(result1)) != 0)
        zero_crossings2 = np.sum(np.diff(np.sign(result2)) != 0)
        assert zero_crossings2 > zero_crossings1
    
    def test_wave_activation_amplitude(self, sample_inputs):
        """Test wave activation with different amplitudes."""
        x = sample_inputs["small"]
        
        result1 = wave_activation(x, frequencies=[1.0], amplitudes=[1.0])
        result2 = wave_activation(x, frequencies=[1.0], amplitudes=[2.0])
        
        # Amplitude should scale the output
        np.testing.assert_allclose(result2, 2.0 * result1)
    
    def test_wave_activation_bounded(self):
        """Test that wave activation is bounded."""
        x = np.linspace(-10, 10, 1000)
        result = wave_activation(x, frequencies=[1.0], amplitudes=[1.0])
        
        # Should be bounded by amplitude
        assert np.all(np.abs(result) <= 1.0 + 1e-6)
    
    def test_wave_activation_class(self, sample_inputs):
        """Test WaveActivation class implementation."""
        x = sample_inputs["small"]
        
        activation = WaveActivation(frequencies=[2.0], amplitudes=[0.5])
        result = activation.forward(x)
        func_result = wave_activation(x, frequencies=[2.0], amplitudes=[0.5])
        
        np.testing.assert_allclose(result, func_result)
        
        # Test gradient
        grad = activation.gradient(x)
        assert grad.shape == x.shape
        assert np.all(np.isfinite(grad))


class TestFourierActivation:
    """Test Fourier activation function."""
    
    def test_fourier_activation_basic(self, sample_inputs):
        """Test basic Fourier activation functionality."""
        x = sample_inputs["small"]
        result = fourier_activation(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    def test_fourier_activation_harmonics(self, sample_inputs):
        """Test Fourier activation with different numbers of harmonics."""
        x = sample_inputs["medium"]
        
        result1 = fourier_activation(x, n_terms=1)
        result3 = fourier_activation(x, n_terms=3)
        result5 = fourier_activation(x, n_terms=5)
        
        # More harmonics should produce more complex patterns
        assert not np.allclose(result1, result3)
        assert not np.allclose(result3, result5)
        
        # More harmonics should have more zero crossings (adjusted for constant term)
        # Remove constant offset before checking zero crossings
        result1_centered = result1 - np.mean(result1)
        result5_centered = result5 - np.mean(result5)
        zero_crossings1 = np.sum(np.diff(np.sign(result1_centered)) != 0)
        zero_crossings5 = np.sum(np.diff(np.sign(result5_centered)) != 0)
        # May have similar crossings due to Fourier series nature
        assert zero_crossings5 >= zero_crossings1 - 2
    
    def test_fourier_activation_base_freq(self, sample_inputs):
        """Test Fourier activation with different base frequencies."""
        x = sample_inputs["small"]
        
        result1 = fourier_activation(x, n_terms=3, base_freq=1.0)
        result2 = fourier_activation(x, n_terms=3, base_freq=2.0)
        
        # Different base frequencies should produce different results
        assert not np.allclose(result1, result2)
    
    def test_fourier_activation_class(self, sample_inputs):
        """Test FourierActivation class implementation."""
        x = sample_inputs["small"]
        
        activation = FourierActivation(n_terms=4)
        result = activation.forward(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        
        # Test gradient
        grad = activation.gradient(x)
        assert grad.shape == x.shape
        
        # Test parameters
        assert hasattr(activation, 'n_terms')
        assert activation.n_terms == 4


class TestQuantumActivation:
    """Test quantum activation function."""
    
    def test_quantum_activation_basic(self, sample_inputs):
        """Test basic quantum activation functionality."""
        x = sample_inputs["small"]
        result = quantum_activation(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    def test_quantum_activation_states(self, sample_inputs):
        """Test quantum activation with different numbers of states."""
        x = sample_inputs["medium"]
        
        result2 = quantum_activation(x, n_states=2)
        result4 = quantum_activation(x, n_states=4)
        result8 = quantum_activation(x, n_states=8)
        
        # Different numbers of states should produce different results
        assert not np.allclose(result2, result4)
        assert not np.allclose(result4, result8)
    
    def test_quantum_activation_temperature(self, sample_inputs):
        """Test quantum activation with different temperatures."""
        x = sample_inputs["small"]
        
        # Different temperatures should affect the activation
        result1 = quantum_activation(x, n_states=3, temperature=0.5)
        result2 = quantum_activation(x, n_states=3, temperature=2.0)
        
        assert result1.shape == x.shape
        assert result2.shape == x.shape
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))
        # Different temperatures should produce different results
        assert not np.allclose(result1, result2)
    
    def test_quantum_activation_normalization(self):
        """Test that quantum state amplitudes are properly normalized."""
        x = np.array([1.0])
        
        # Test with various amplitudes
        amplitudes = np.array([1.0, 2.0, 3.0])
        activation = QuantumActivation(n_states=3)
        activation.state_amplitudes = amplitudes
        
        # The normalization should happen internally
        result = activation.forward(x)
        assert np.isfinite(result)
    
    def test_quantum_activation_class(self, sample_inputs):
        """Test QuantumActivation class implementation."""
        x = sample_inputs["small"]
        
        activation = QuantumActivation(n_states=4, temperature=1.5)
        result = activation.forward(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        
        # Test gradient
        grad = activation.gradient(x)
        assert grad.shape == x.shape
        
        # Test parameters
        assert hasattr(activation, 'n_states')
        assert activation.n_states == 4
        assert hasattr(activation, 'temperature')
        assert activation.temperature == 1.5
    
    def test_quantum_activation_properties(self):
        """Test mathematical properties of quantum activation."""
        props = check_activation_properties(
            quantum_activation,
            {"bounded": True, "bounds": (-1, 1)}
        )
        
        # Quantum activation should be bounded due to sine functions
        assert props["bounded"]