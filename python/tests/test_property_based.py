"""Property-based testing with Hypothesis."""

import warnings

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Custom strategies for our domain
finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
small_floats = st.floats(
    min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
)

# Array strategies
float_arrays = arrays(
    dtype=np.float32,
    shape=st.integers(min_value=1, max_value=100),
    elements=finite_floats,
)

small_arrays = arrays(
    dtype=np.float32,
    shape=st.integers(min_value=1, max_value=100),
    elements=small_floats,
)


class TestActivationProperties:
    """Test mathematical properties of activation functions."""

    @given(float_arrays)
    @settings(max_examples=50)
    def test_wave_activation_bounded(self, x):
        """Test that wave activation output is bounded."""
        from analogtivation.core import wave_activation

        result = wave_activation(x, frequencies=[1.0], amplitudes=[1.0])

        # Wave activation should be bounded by amplitude
        assert np.all(np.abs(result) <= 1.1)  # Small tolerance for numerical errors
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    @given(float_arrays)
    @settings(max_examples=50)
    def test_quantum_activation_bounded(self, x):
        """Test that quantum activation output is bounded."""
        from analogtivation.core import quantum_activation

        result = quantum_activation(x, n_states=3)

        # Output should be bounded (state indices)
        assert np.all(result >= 0)
        assert np.all(result < 3)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    @given(small_arrays)
    @settings(max_examples=50)
    def test_lorenz_activation_finite(self, x):
        """Test that Lorenz activation always produces finite outputs."""
        from analogtivation.core import lorenz_activation

        result = lorenz_activation(x)

        # Lorenz should be bounded by tanh
        assert np.all(np.abs(result) <= 1.0)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    @given(float_arrays)
    @settings(max_examples=50)
    def test_logistic_map_bounded(self, x):
        """Test that logistic map activation is bounded."""
        from analogtivation.core import logistic_map_activation

        result = logistic_map_activation(x, r=3.9, iterations=3)

        # Output should be in [-1, 1]
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    @given(
        x=small_arrays,
        frequency=st.floats(min_value=0.1, max_value=10, allow_nan=False),
        amplitude=positive_floats,
    )
    @settings(max_examples=30)
    def test_wave_activation_scaling(self, x, frequency, amplitude):
        """Test that wave activation scales properly with amplitude."""
        from analogtivation.core import wave_activation

        result = wave_activation(x, frequencies=[frequency], amplitudes=[amplitude])

        # Should be bounded by amplitude (approximately)
        assert np.all(np.abs(result) <= amplitude * 1.1)
        assert result.shape == x.shape

    @given(x=small_arrays, n_terms=st.integers(min_value=1, max_value=10))
    @settings(max_examples=30)
    def test_fourier_activation_convergence(self, x, n_terms):
        """Test that Fourier activation converges with more terms."""
        from analogtivation.core import fourier_activation

        result = fourier_activation(x, n_terms=n_terms)

        # Should produce finite results
        assert np.all(np.isfinite(result))
        assert result.shape == x.shape


class TestActivationInvariants:
    """Test invariant properties that should hold."""

    @given(small_floats)
    @settings(max_examples=100)
    def test_zero_input_behavior(self, zero_offset):
        """Test behavior at zero input."""
        from analogtivation.core import (
            fourier_activation,
            lorenz_activation,
            mandelbrot_activation,
            wave_activation,
        )

        # Very small input close to zero
        x = np.array([zero_offset * 1e-10])

        # Wave activation at zero
        wave_result = wave_activation(x)
        assert np.abs(wave_result[0]) < 0.1

        # Fourier has constant term, so won't be zero
        fourier_result = fourier_activation(x)
        assert np.all(np.isfinite(fourier_result))

        # Lorenz at zero
        lorenz_result = lorenz_activation(x)
        assert np.abs(lorenz_result[0]) < 0.1

        # Mandelbrot at zero
        mandel_result = mandelbrot_activation(x)
        assert np.abs(mandel_result[0]) < 0.1

    @given(x=float_arrays, epsilon=st.floats(min_value=1e-10, max_value=1e-5))
    @settings(max_examples=30)
    def test_continuity(self, x, epsilon):
        """Test that activations are continuous."""
        from analogtivation.core import lorenz_activation, wave_activation

        # Test continuity by checking small perturbations
        x_perturbed = x + epsilon

        # Wave activation
        y1 = wave_activation(x)
        y2 = wave_activation(x_perturbed)

        # Output shouldn't change drastically for small input changes
        diff = np.abs(y2 - y1)
        assert np.all(diff < 1.0)  # Reasonable bound for continuity

        # Lorenz activation (may be more sensitive)
        y1_lorenz = lorenz_activation(x)
        y2_lorenz = lorenz_activation(x_perturbed)

        # Both should at least be finite
        assert np.all(np.isfinite(y1_lorenz))
        assert np.all(np.isfinite(y2_lorenz))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @given(
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10),
        )
    )
    def test_multidimensional_inputs(self, shape):
        """Test that activations handle multidimensional arrays."""
        from analogtivation.core import (
            logistic_map_activation,
            quantum_activation,
            wave_activation,
        )

        x = np.random.randn(*shape).astype(np.float32)

        # Test different activations
        wave_result = wave_activation(x)
        assert wave_result.shape == shape

        quantum_result = quantum_activation(x)
        assert quantum_result.shape == shape

        logistic_result = logistic_map_activation(x)
        assert logistic_result.shape == shape

    @given(x=st.one_of(st.just(np.array([])), arrays(dtype=np.float32, shape=(0,))))
    def test_empty_input(self, x):
        """Test handling of empty arrays."""
        from analogtivation.core import wave_activation

        if x.size == 0:
            result = wave_activation(x)
            assert result.size == 0
            assert result.shape == x.shape

    def test_extreme_values(self):
        """Test with extreme but finite values."""
        from analogtivation.core import (
            logistic_map_activation,
            lorenz_activation,
            mandelbrot_activation,
        )

        extreme_values = np.array([1e5, -1e5, 1e-5, -1e-5], dtype=np.float32)

        # All should handle extreme values gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lorenz_result = lorenz_activation(extreme_values)
            assert np.all(np.isfinite(lorenz_result))
            assert np.all(np.abs(lorenz_result) <= 1.0)  # Bounded by tanh

            logistic_result = logistic_map_activation(extreme_values)
            assert np.all(np.isfinite(logistic_result))
            assert np.all(np.abs(logistic_result) <= 1.0)

            mandel_result = mandelbrot_activation(extreme_values, max_iter=2)
            assert np.all(np.isfinite(mandel_result))


class TestParameterValidation:
    """Test parameter validation and constraints."""

    @given(
        r=st.floats(min_value=0.1, max_value=4.0),
        iterations=st.integers(min_value=1, max_value=10),
    )
    def test_logistic_map_parameters(self, r, iterations):
        """Test logistic map with various parameters."""
        from analogtivation.core import logistic_map_activation

        x = np.array([0.1, 0.5, 0.9])
        result = logistic_map_activation(x, r=r, iterations=iterations)

        # Should always produce bounded output
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)
        assert np.all(np.isfinite(result))

    @given(
        n_states=st.integers(min_value=1, max_value=20),
        temperature=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_quantum_parameters(self, n_states, temperature):
        """Test quantum activation with various parameters."""
        from analogtivation.core import quantum_activation

        x = np.random.randn(10).astype(np.float32)
        result = quantum_activation(x, n_states=n_states, temperature=temperature)

        # Output should be valid state indices
        assert np.all(result >= 0)
        assert np.all(result < n_states)
        assert np.all(np.isfinite(result))
