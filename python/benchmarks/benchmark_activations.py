"""Performance benchmarks for activation functions."""

from functools import partial

import numpy as np
import pytest


class BenchmarkBase:
    """Base class for activation benchmarks."""

    def setup(self):
        """Set up test data."""
        self.sizes = [100, 1000, 10000, 100000]
        self.data = {
            size: np.random.randn(size).astype(np.float32) for size in self.sizes
        }
        self.data_2d = {
            size: np.random.randn(size, 32).astype(np.float32) for size in self.sizes
        }


class TestCoreActivationBenchmarks(BenchmarkBase):
    """Benchmark core NumPy implementations."""

    def test_clock_activation_performance(self, benchmark):
        """Benchmark clock activation."""
        from analogtivation.core import clock_activation

        # Benchmark with different sizes
        for size in [1000, 10000]:
            data = self.data[size]
            result = benchmark.pedantic(
                clock_activation, args=(data,), iterations=10, rounds=5
            )
            assert result.shape == data.shape

    def test_wave_activation_performance(self, benchmark):
        """Benchmark wave activation."""
        from analogtivation.core import wave_activation

        data = self.data[10000]
        result = benchmark(wave_activation, data)
        assert result.shape == data.shape

    def test_fourier_activation_performance(self, benchmark):
        """Benchmark Fourier activation."""
        from analogtivation.core import fourier_activation

        data = self.data[10000]
        result = benchmark(fourier_activation, data, n_terms=5)
        assert result.shape == data.shape

    def test_quantum_activation_performance(self, benchmark):
        """Benchmark quantum activation."""
        from analogtivation.core import quantum_activation

        data = self.data[10000]
        result = benchmark(quantum_activation, data, n_states=3)
        assert result.shape == data.shape

    def test_lorenz_activation_performance(self, benchmark):
        """Benchmark Lorenz activation."""
        from analogtivation.core import lorenz_activation

        data = self.data[10000]
        result = benchmark(lorenz_activation, data)
        assert result.shape == data.shape

    def test_logistic_map_performance(self, benchmark):
        """Benchmark logistic map activation."""
        from analogtivation.core import logistic_map_activation

        data = self.data[10000]
        result = benchmark(logistic_map_activation, data, iterations=3)
        assert result.shape == data.shape

    def test_mandelbrot_activation_performance(self, benchmark):
        """Benchmark Mandelbrot activation."""
        from analogtivation.core import mandelbrot_activation

        data = self.data[10000]
        result = benchmark(mandelbrot_activation, data, max_iter=3)
        assert result.shape == data.shape


@pytest.mark.skipif("not config.getoption('--benchmark-only')")
class TestFrameworkActivationBenchmarks(BenchmarkBase):
    """Benchmark framework-specific implementations."""

    def test_tensorflow_activations(self, benchmark, skip_if_no_tensorflow):
        """Benchmark TensorFlow activations."""
        import tensorflow as tf

        from analogtivation.tensorflow import (
            FourierActivation,
            LorenzActivation,
            WaveActivation,
        )

        # Test data
        data = tf.constant(self.data_2d[10000])

        # Benchmark different activations
        activations = [
            WaveActivation(),
            FourierActivation(n_harmonics=5),
            LorenzActivation(),
        ]

        for activation in activations:
            result = benchmark(activation, data)
            assert result.shape == data.shape

    def test_pytorch_activations(self, benchmark, skip_if_no_torch):
        """Benchmark PyTorch activations."""
        import torch

        from analogtivation.torch import (
            FourierActivation,
            LorenzActivation,
            WaveActivation,
        )

        # Test data
        data = torch.tensor(self.data_2d[10000])

        # Benchmark different activations
        activations = [
            WaveActivation(),
            FourierActivation(n_harmonics=5),
            LorenzActivation(),
        ]

        for activation in activations:
            result = benchmark(activation, data)
            assert result.shape == data.shape

    def test_jax_activations(self, benchmark, skip_if_no_jax):
        """Benchmark JAX activations."""
        import jax.numpy as jnp

        from analogtivation.jax import fourier_activation, wave_activation

        # Test data
        data = jnp.array(self.data_2d[10000])

        # Benchmark different activations
        # Note: First call includes JIT compilation time
        benchmark(wave_activation, data)

        # Pre-JIT compile for fair comparison
        weights = jnp.ones(5)
        fourier_jit = partial(fourier_activation, weights=weights, n_harmonics=5)
        fourier_jit(data[:10])  # Warm up JIT

        result = benchmark(fourier_jit, data)
        assert result.shape == data.shape


class TestActivationComparison(BenchmarkBase):
    """Compare performance across different activation types."""

    @pytest.mark.parametrize("size", [1000, 10000])
    def test_compare_all_activations(self, benchmark, size):
        """Compare all activation functions at given size."""
        from analogtivation.core import (
            clock_activation,
            fourier_activation,
            logistic_map_activation,
            lorenz_activation,
            mandelbrot_activation,
            quantum_activation,
            wave_activation,
        )

        data = self.data[size]

        activations = {
            "clock": clock_activation,
            "wave": wave_activation,
            "fourier": partial(fourier_activation, n_terms=5),
            "quantum": partial(quantum_activation, n_states=3),
            "lorenz": lorenz_activation,
            "logistic_map": partial(logistic_map_activation, iterations=3),
            "mandelbrot": partial(mandelbrot_activation, max_iter=3),
        }

        results = {}
        for name, activation in activations.items():
            benchmark.name = f"{name}_{size}"
            results[name] = benchmark(activation, data)

        # Verify all produced valid results
        for name, result in results.items():
            assert result.shape == data.shape
            assert np.all(np.isfinite(result))


class TestMemoryUsage:
    """Test memory usage of activation functions."""

    def test_memory_efficiency(self):
        """Test that activations don't use excessive memory."""
        import tracemalloc

        from analogtivation.core import (
            fourier_activation,
            quantum_activation,
            wave_activation,
        )

        # Large input
        data = np.random.randn(1000000).astype(np.float32)

        # Measure memory usage
        activations = [
            ("wave", wave_activation),
            ("fourier", lambda x: fourier_activation(x, n_terms=10)),
            ("quantum", lambda x: quantum_activation(x, n_states=5)),
        ]

        for name, activation in activations:
            tracemalloc.start()

            # Run activation
            result = activation(data)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Memory usage should be reasonable (less than 10x input size)
            input_memory = data.nbytes
            assert peak < input_memory * 10, f"{name} uses too much memory"

            # Result should be valid
            assert np.all(np.isfinite(result))
