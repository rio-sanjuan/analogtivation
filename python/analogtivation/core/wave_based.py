"""Wave-based activation functions."""

import numpy as np
from typing import Union
from ..base import ActivationFunction


class WaveActivation(ActivationFunction):
    """
    Composite waveform activation function.
    
    Combines multiple sinusoidal waves with different frequencies.
    """
    
    def __init__(self, frequencies: list = [1.0, 2.0, 3.0], 
                 amplitudes: list = [1.0, 0.5, 0.25]):
        """
        Initialize wave activation.
        
        Parameters
        ----------
        frequencies : list
            List of frequencies for component waves
        amplitudes : list
            List of amplitudes for component waves
        """
        super().__init__("wave", frequencies=frequencies, amplitudes=amplitudes)
        self.frequencies = np.array(frequencies)
        self.amplitudes = np.array(amplitudes)
        
    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply composite wave activation."""
        x_array = np.asarray(x)
        result = np.zeros_like(x_array, dtype=np.float64)
        
        for freq, amp in zip(self.frequencies, self.amplitudes):
            result += amp * np.sin(2 * np.pi * freq * x_array)
            
        return result if isinstance(x, np.ndarray) else float(result)
    
    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of wave activation."""
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
            grad -= amplitude * n * self.base_freq * np.pi * np.sin(n * self.base_freq * np.pi * x_array)
            grad += amplitude * n * self.base_freq * np.pi * np.cos(n * self.base_freq * np.pi * x_array)
            
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
        
        grad = (self.forward(x_array + epsilon) - self.forward(x_array - epsilon)) / (2 * epsilon)
        
        return grad if isinstance(x, np.ndarray) else float(grad)


# Convenience functions for functional API
def wave_activation(x: Union[np.ndarray, float], 
                   frequencies: list = [1.0, 2.0, 3.0],
                   amplitudes: list = [1.0, 0.5, 0.25]) -> Union[np.ndarray, float]:
    """Functional interface for wave activation."""
    return WaveActivation(frequencies, amplitudes).forward(x)


def fourier_activation(x: Union[np.ndarray, float],
                      n_terms: int = 5,
                      base_freq: float = 1.0) -> Union[np.ndarray, float]:
    """Functional interface for Fourier activation."""
    return FourierActivation(n_terms, base_freq).forward(x)


def quantum_activation(x: Union[np.ndarray, float],
                      n_states: int = 3,
                      temperature: float = 1.0) -> Union[np.ndarray, float]:
    """Functional interface for quantum activation."""
    return QuantumActivation(n_states, temperature).forward(x)