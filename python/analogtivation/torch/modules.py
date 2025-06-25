"""PyTorch module implementations of creative activation functions."""

import torch
import torch.nn as nn
from typing import Optional
from .functional import (
    clock_activation,
    seasonal_activation,
    circadian_activation,
    wave_activation,
    fourier_activation,
    quantum_activation,
    lorenz_activation,
    logistic_map_activation,
    mandelbrot_activation,
)


class ClockActivation(nn.Module):
    """Clock activation module that changes behavior based on current time."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return clock_activation(input)
    
    def extra_repr(self) -> str:
        return 'time-dependent'


class SeasonalActivation(nn.Module):
    """Seasonal activation module that varies with seasons."""
    
    def __init__(self, hemisphere: str = "northern"):
        super().__init__()
        self.hemisphere = hemisphere
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return seasonal_activation(input, self.hemisphere)
    
    def extra_repr(self) -> str:
        return f'hemisphere={self.hemisphere}'


class CircadianActivation(nn.Module):
    """24-hour circadian rhythm activation module."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return circadian_activation(input)
    
    def extra_repr(self) -> str:
        return '24-hour cycle'


class WaveActivation(nn.Module):
    """Composite waveform activation module."""
    
    def __init__(self, frequency: float = 1.0, amplitude: float = 1.0):
        super().__init__()
        self.frequency = frequency
        self.amplitude = amplitude
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return wave_activation(input, self.frequency, self.amplitude)
    
    def extra_repr(self) -> str:
        return f'frequency={self.frequency}, amplitude={self.amplitude}'


class FourierActivation(nn.Module):
    """Fourier series-based activation module."""
    
    def __init__(self, n_harmonics: int = 3):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.harmonic_weights = nn.Parameter(torch.ones(n_harmonics))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return fourier_activation(input, self.harmonic_weights, self.n_harmonics)
    
    def extra_repr(self) -> str:
        return f'n_harmonics={self.n_harmonics}'


class QuantumActivation(nn.Module):
    """Quantum-inspired activation module with superposition."""
    
    def __init__(self, n_states: int = 2):
        super().__init__()
        self.n_states = n_states
        self.state_amplitudes = nn.Parameter(torch.randn(n_states) * 0.1)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantum_activation(input, self.state_amplitudes, self.n_states)
    
    def extra_repr(self) -> str:
        return f'n_states={self.n_states}'


class LorenzActivation(nn.Module):
    """Lorenz attractor-inspired activation module."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return lorenz_activation(input, self.sigma, self.rho, self.beta)
    
    def extra_repr(self) -> str:
        return f'sigma={self.sigma}, rho={self.rho}, beta={self.beta}'


class LogisticMapActivation(nn.Module):
    """Logistic map chaos activation module."""
    
    def __init__(self, r: float = 3.9, iterations: int = 3):
        super().__init__()
        self.r = r
        self.iterations = iterations
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return logistic_map_activation(input, self.r, self.iterations)
    
    def extra_repr(self) -> str:
        return f'r={self.r}, iterations={self.iterations}'


class MandelbrotActivation(nn.Module):
    """Mandelbrot set-inspired activation module."""
    
    def __init__(self, max_iterations: int = 5):
        super().__init__()
        self.max_iterations = max_iterations
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return mandelbrot_activation(input, self.max_iterations)
    
    def extra_repr(self) -> str:
        return f'max_iterations={self.max_iterations}'