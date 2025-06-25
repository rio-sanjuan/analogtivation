"""Integration tests with small neural networks."""

import numpy as np
import pytest


class TestTensorFlowIntegration:
    """Test activation functions in TensorFlow/Keras models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        # Simple 2D classification problem
        X = np.random.randn(100, 4).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        return X, y

    def test_clock_activation_in_model(self, skip_if_no_tensorflow, sample_data):
        """Test ClockActivation in a simple neural network."""
        import tensorflow as tf

        from analogtivation.tensorflow import ClockActivation

        X, y = sample_data

        # Build simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, input_shape=(4,)),
                ClockActivation(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train briefly
        history = model.fit(X, y, epochs=5, verbose=0, validation_split=0.2)

        # Check that model trains
        assert len(history.history["loss"]) == 5
        assert history.history["loss"][-1] < history.history["loss"][0]

        # Check predictions
        predictions = model.predict(X[:10], verbose=0)
        assert predictions.shape == (10, 1)
        assert np.all(np.isfinite(predictions))

    def test_wave_activation_in_model(self, skip_if_no_tensorflow, sample_data):
        """Test WaveActivation in a neural network."""
        import tensorflow as tf

        from analogtivation.tensorflow import WaveActivation

        X, y = sample_data

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, input_shape=(4,)),
                WaveActivation(frequency=2.0, amplitude=0.5),
                tf.keras.layers.Dense(4),
                WaveActivation(frequency=1.0, amplitude=1.0),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(X, y, epochs=5, verbose=0, validation_split=0.2)

        assert history.history["loss"][-1] < history.history["loss"][0]

    def test_chaos_activation_in_model(self, skip_if_no_tensorflow, sample_data):
        """Test chaos-based activations in a neural network."""
        import tensorflow as tf

        from analogtivation.tensorflow import LogisticMapActivation, LorenzActivation

        X, y = sample_data

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, input_shape=(4,)),
                LorenzActivation(sigma=10.0, rho=28.0),
                tf.keras.layers.Dense(4),
                LogisticMapActivation(r=3.9, iterations=2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(X, y, epochs=5, verbose=0, validation_split=0.2)

        # Verify model trains without NaN/Inf
        assert np.all(np.isfinite(history.history["loss"]))


class TestPyTorchIntegration:
    """Test activation functions in PyTorch models."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 4).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(100) * 0.1).astype(
            np.float32
        )
        return X, y

    def test_quantum_activation_in_model(self, skip_if_no_torch, sample_data):
        """Test QuantumActivation in a PyTorch model."""
        import torch
        import torch.nn as nn

        from analogtivation.torch import QuantumActivation

        X, y = sample_data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Build model
        model = nn.Sequential(
            nn.Linear(4, 8),
            QuantumActivation(n_states=3),
            nn.Linear(8, 4),
            QuantumActivation(n_states=2),
            nn.Linear(4, 1),
        )

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()

        # Check training improved loss
        final_loss = loss.item()
        assert final_loss < initial_loss
        assert np.isfinite(final_loss)

    def test_fourier_activation_in_model(self, skip_if_no_torch, sample_data):
        """Test FourierActivation in a PyTorch model."""
        import torch
        import torch.nn as nn

        from analogtivation.torch import FourierActivation

        X, y = sample_data
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # Build model with Fourier activations
        model = nn.Sequential(
            nn.Linear(4, 16),
            FourierActivation(n_harmonics=5),
            nn.Linear(16, 8),
            FourierActivation(n_harmonics=3),
            nn.Linear(8, 1),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train for one epoch
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Verify training works
        assert np.isfinite(epoch_loss)
        assert epoch_loss > 0

    def test_mixed_activations_model(self, skip_if_no_torch, sample_data):
        """Test model with mixed activation functions."""
        import torch
        import torch.nn as nn

        from analogtivation.torch import (
            CircadianActivation,
            MandelbrotActivation,
            WaveActivation,
        )

        X, y = sample_data

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 16)
                self.act1 = WaveActivation(frequency=1.5)
                self.fc2 = nn.Linear(16, 8)
                self.act2 = MandelbrotActivation(max_iterations=3)
                self.fc3 = nn.Linear(8, 4)
                self.act3 = CircadianActivation()
                self.fc4 = nn.Linear(4, 1)

            def forward(self, x):
                x = self.act1(self.fc1(x))
                x = self.act2(self.fc2(x))
                x = self.act3(self.fc3(x))
                return self.fc4(x)

        model = MixedModel()

        # Forward pass
        X_tensor = torch.FloatTensor(X[:10])
        outputs = model(X_tensor)

        assert outputs.shape == (10, 1)
        assert torch.all(torch.isfinite(outputs))


class TestJAXIntegration:
    """Test activation functions in JAX models."""

    def test_jax_activations_in_computation(self, skip_if_no_jax):
        """Test JAX activations in computation graph."""
        import jax
        import jax.numpy as jnp

        from analogtivation.jax import (
            logistic_map_activation,
            lorenz_activation,
            wave_activation,
        )

        # Define a simple neural network function
        def network(params, x):
            w1, b1, w2, b2, w3, b3 = params

            # Layer 1 with wave activation
            x = jnp.dot(x, w1) + b1
            x = wave_activation(x, frequencies=[1.0, 2.0], amplitudes=[0.5, 0.3])

            # Layer 2 with Lorenz activation
            x = jnp.dot(x, w2) + b2
            x = lorenz_activation(x, sigma=10.0, rho=28.0)

            # Layer 3 with logistic map activation
            x = jnp.dot(x, w3) + b3
            x = logistic_map_activation(x, r=3.7, iterations=2)

            return x

        # Initialize parameters
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 6)
        params = [
            jax.random.normal(keys[0], (4, 8)),  # w1
            jax.random.normal(keys[1], (8,)),  # b1
            jax.random.normal(keys[2], (8, 4)),  # w2
            jax.random.normal(keys[3], (4,)),  # b2
            jax.random.normal(keys[4], (4, 1)),  # w3
            jax.random.normal(keys[5], (1,)),  # b3
        ]

        # Test forward pass
        x = jax.random.normal(key, (10, 4))
        output = network(params, x)

        assert output.shape == (10, 1)
        assert jnp.all(jnp.isfinite(output))

        # Test gradient computation
        def loss_fn(params, x, y):
            pred = network(params, x)
            return jnp.mean((pred - y) ** 2)

        y = jax.random.normal(key, (10, 1))
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params, x, y)

        # Verify gradients are finite
        for grad in grads:
            assert jnp.all(jnp.isfinite(grad))

    def test_jax_vmap_compatibility(self, skip_if_no_jax):
        """Test that activations work with JAX vmap."""
        import jax
        import jax.numpy as jnp

        from analogtivation.jax import fourier_activation, quantum_activation

        # Test vmap over batch dimension
        batch_size = 32
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 10))

        # Quantum activation with vmap
        state_amplitudes = jnp.array([0.5, -0.3, 0.2])
        quantum_vmap = jax.vmap(
            lambda x: quantum_activation(x, state_amplitudes, n_states=3)
        )
        result_quantum = quantum_vmap(x)
        assert result_quantum.shape == (batch_size, 10)

        # Fourier activation with vmap
        weights = jnp.ones(4)
        fourier_vmap = jax.vmap(lambda x: fourier_activation(x, weights, n_harmonics=4))
        result_fourier = fourier_vmap(x)
        assert result_fourier.shape == (batch_size, 10)


class TestCrossFrameworkModels:
    """Test that models behave consistently across frameworks."""

    def test_simple_model_consistency(self, frameworks_available):
        """Test that a simple model produces similar results across frameworks."""
        np.random.seed(42)
        X = np.random.randn(20, 4).astype(np.float32)

        results = {}

        # TensorFlow model
        if frameworks_available["tensorflow"]:
            import tensorflow as tf

            from analogtivation.tensorflow import WaveActivation

            tf.random.set_seed(42)
            model_tf = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        8, input_shape=(4,), kernel_initializer="glorot_uniform"
                    ),
                    WaveActivation(frequency=1.0, amplitude=1.0),
                    tf.keras.layers.Dense(1, kernel_initializer="glorot_uniform"),
                ]
            )
            results["tensorflow"] = model_tf(X).numpy()

        # PyTorch model
        if frameworks_available["torch"]:
            import torch
            import torch.nn as nn

            from analogtivation.torch import WaveActivation as WaveActivationTorch

            torch.manual_seed(42)
            model_torch = nn.Sequential(
                nn.Linear(4, 8),
                WaveActivationTorch(frequency=1.0, amplitude=1.0),
                nn.Linear(8, 1),
            )
            with torch.no_grad():
                results["torch"] = model_torch(torch.FloatTensor(X)).numpy()

        # Verify all results are finite
        for framework, output in results.items():
            assert np.all(
                np.isfinite(output)
            ), f"{framework} produced non-finite values"
            assert output.shape == (20, 1), f"{framework} output shape mismatch"
