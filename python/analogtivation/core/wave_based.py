"""Wave-based activation functions.

This module provides activation functions based on wave physics and
signal processing concepts. These functions create complex, periodic
transformations useful for:

- Modeling periodic or oscillatory phenomena
- Feature extraction at multiple scales
- Creating learnable frequency-domain representations
- Signal processing and spectral analysis tasks

The wave-based activations can capture multi-scale patterns and
harmonic relationships in data.

Examples
--------
>>> import numpy as np
>>> from analogtivation.core import wave_activation, fourier_activation
>>> x = np.linspace(-2*np.pi, 2*np.pi, 100)
>>> # Single frequency wave
>>> y1 = wave_activation(x, frequencies=[1.0], amplitudes=[1.0])
>>> # Multi-frequency composite
>>> y2 = wave_activation(x, frequencies=[1, 2, 3], amplitudes=[1, 0.5, 0.25])
"""

from typing import Union

import numpy as np

from ..base import ActivationFunction


class WaveActivation(ActivationFunction):
    """
    Composite waveform activation function.

    Combines multiple sinusoidal waves with different frequencies and
    amplitudes to create complex periodic patterns. This activation can
    learn to represent multi-scale features and harmonic relationships.

    The output is a weighted sum of sine waves:
    f(x) = sum(A_i * sin(2π * f_i * x)) for each frequency f_i and amplitude A_i

    Attributes
    ----------
    frequencies : np.ndarray
        Array of frequency values for component waves
    amplitudes : np.ndarray
        Array of amplitude values for component waves
    """

    def __init__(
        self, frequencies: list = [1.0, 2.0, 3.0], amplitudes: list = [1.0, 0.5, 0.25]
    ):
        """
        Initialize wave activation.

        Parameters
        ----------
        frequencies : list, optional
            List of frequencies for component waves. Higher frequencies
            capture finer details. Default is [1.0, 2.0, 3.0].
        amplitudes : list, optional
            List of amplitudes for component waves. Should have same
            length as frequencies. Default is [1.0, 0.5, 0.25].

        Raises
        ------
        ValueError
            If frequencies and amplitudes have different lengths

        Examples
        --------
        >>> # Simple single-frequency wave
        >>> act1 = WaveActivation(frequencies=[1.0], amplitudes=[1.0])
        >>> # Harmonic series with decaying amplitudes
        >>> act2 = WaveActivation(
        ...     frequencies=[1, 2, 3, 4],
        ...     amplitudes=[1, 0.5, 0.33, 0.25]
        ... )
        """
        super().__init__("wave", frequencies=frequencies, amplitudes=amplitudes)
        self.frequencies = np.array(frequencies)
        self.amplitudes = np.array(amplitudes)

    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply composite wave activation.

        Parameters
        ----------
        x : array_like
            Input values

        Returns
        -------
        array_like
            Sum of sinusoidal components with specified frequencies
            and amplitudes
        """
        x_array = np.asarray(x)
        result = np.zeros_like(x_array, dtype=np.float64)

        for freq, amp in zip(self.frequencies, self.amplitudes):
            result += amp * np.sin(2 * np.pi * freq * x_array)

        return result if isinstance(x, np.ndarray) else float(result)

    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of wave activation.

        The gradient is the sum of cosine terms:
        f'(x) = sum(A_i * 2π * f_i * cos(2π * f_i * x))

        Parameters
        ----------
        x : array_like
            Input values

        Returns
        -------
        array_like
            Gradient values with same shape as input
        """
        x_array = np.asarray(x)
        grad = np.zeros_like(x_array, dtype=np.float64)

        for freq, amp in zip(self.frequencies, self.amplitudes):
            grad += amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * x_array)

        return grad if isinstance(x, np.ndarray) else float(grad)


class FourierActivation(ActivationFunction):
    """
    Fourier series based activation function.

    Uses truncated Fourier series to create complex activation patterns.
    """

    def __init__(self, n_terms: int = 5, base_freq: float = 1.0):
        """
        Initialize Fourier activation.

        Parameters
        ----------
        n_terms : int
            Number of Fourier terms
        base_freq : float
            Base frequency
        """
        super().__init__("fourier", n_terms=n_terms, base_freq=base_freq)
        self.n_terms = n_terms
        self.base_freq = base_freq

    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply Fourier activation."""
        x_array = np.asarray(x)
        result = np.zeros_like(x_array, dtype=np.float64)

        # Add constant term
        result += 0.5

        # Add cosine and sine terms
        for n in range(1, self.n_terms + 1):
            # Decreasing amplitude with frequency
            amplitude = 1.0 / n
            result += amplitude * np.cos(n * self.base_freq * np.pi * x_array)
            result += amplitude * np.sin(n * self.base_freq * np.pi * x_array)

        return result if isinstance(x, np.ndarray) else float(result)

    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of Fourier activation."""
        x_array = np.asarray(x)
        grad = np.zeros_like(x_array, dtype=np.float64)

        for n in range(1, self.n_terms + 1):
            amplitude = 1.0 / n
            grad -= (
                amplitude
                * n
                * self.base_freq
                * np.pi
                * np.sin(n * self.base_freq * np.pi * x_array)
            )
            grad += (
                amplitude
                * n
                * self.base_freq
                * np.pi
                * np.cos(n * self.base_freq * np.pi * x_array)
            )

        return grad if isinstance(x, np.ndarray) else float(grad)


class QuantumActivation(ActivationFunction):
    """
    Quantum-inspired probabilistic activation function.

    Models quantum superposition and measurement collapse.
    """

    def __init__(self, n_states: int = 3, temperature: float = 1.0):
        """
        Initialize quantum activation.

        Parameters
        ----------
        n_states : int
            Number of quantum states
        temperature : float
            Temperature parameter for state transitions
        """
        super().__init__("quantum", n_states=n_states, temperature=temperature)
        self.n_states = n_states
        self.temperature = temperature

    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply quantum activation."""
        x_array = np.asarray(x)

        # Create quantum state amplitudes
        states = np.zeros((self.n_states,) + x_array.shape)
        for i in range(self.n_states):
            # Different energy levels
            energy = i * np.abs(x_array) / self.temperature
            states[i] = np.exp(-energy) * np.cos(np.pi * i * x_array / 2)

        # Normalize probabilities
        probabilities = np.abs(states) ** 2
        probabilities = probabilities / np.sum(probabilities, axis=0, keepdims=True)

        # Collapse to expectation value
        result = np.sum(np.arange(self.n_states).reshape(-1, 1) * probabilities, axis=0)

        return result if isinstance(x, np.ndarray) else float(result)

    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of quantum activation."""
        # Numerical gradient for complex quantum function
        epsilon = 1e-7
        x_array = np.asarray(x)

        grad = (self.forward(x_array + epsilon) - self.forward(x_array - epsilon)) / (
            2 * epsilon
        )

        return grad if isinstance(x, np.ndarray) else float(grad)


# Convenience functions for functional API
def wave_activation(
    x: Union[np.ndarray, float],
    frequencies: list = [1.0, 2.0, 3.0],
    amplitudes: list = [1.0, 0.5, 0.25],
) -> Union[np.ndarray, float]:
    """Functional interface for wave activation."""
    return WaveActivation(frequencies, amplitudes).forward(x)


def fourier_activation(
    x: Union[np.ndarray, float], n_terms: int = 5, base_freq: float = 1.0
) -> Union[np.ndarray, float]:
    """Functional interface for Fourier activation."""
    return FourierActivation(n_terms, base_freq).forward(x)


def quantum_activation(
    x: Union[np.ndarray, float], n_states: int = 3, temperature: float = 1.0
) -> Union[np.ndarray, float]:
    """Functional interface for quantum activation."""
    return QuantumActivation(n_states, temperature).forward(x)
