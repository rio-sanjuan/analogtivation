"""PyTorch functional implementations of creative activation functions."""

import torch
import torch.nn.functional as F
import numpy as np
from math import radians, tan
from time import gmtime, strftime
from typing import Optional


def clock_activation(input: torch.Tensor) -> torch.Tensor:
    """
    Clock activation function that changes behavior based on current time.
    
    Args:
        input: Input tensor
        
    Returns:
        Activated tensor based on current clock time
    """
    current_time = gmtime()
    
    hour = int(strftime("%H", current_time))
    minute = int(strftime("%M", current_time))
    second = int(strftime("%S", current_time))
    
    # Calculate exact positions
    exact_hour = hour % 12 + minute / 60 + second / (60 * 60)
    exact_minute = minute + second / 60
    
    # Convert to angles (clock starts at 12 = 90 degrees)
    hour_hand_angle = -1 * (360 * exact_hour / 12 - 90)
    minute_hand_angle = -1 * (360 * exact_minute / 60 - 90)
    
    # Calculate slopes
    hour_slope = tan(radians(hour_hand_angle))
    minute_slope = tan(radians(minute_hand_angle))
    
    # Apply different slopes based on sign
    return torch.where(input >= 0, minute_slope * input, hour_slope * input)


def seasonal_activation(input: torch.Tensor, hemisphere: str = "northern") -> torch.Tensor:
    """
    Activation function that varies with seasons.
    
    Args:
        input: Input tensor
        hemisphere: Either "northern" or "southern" hemisphere
        
    Returns:
        Seasonally adjusted activation
    """
    current_time = gmtime()
    day_of_year = int(strftime("%j", current_time))
    
    # Adjust for hemisphere
    if hemisphere == "southern":
        day_of_year = (day_of_year + 182) % 365
        
    # Calculate seasonal factor (peaks in summer, troughs in winter)
    seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Apply seasonal modulation
    return input * (1 + 0.3 * seasonal_factor)


def circadian_activation(input: torch.Tensor) -> torch.Tensor:
    """
    24-hour circadian rhythm activation function.
    
    Args:
        input: Input tensor
        
    Returns:
        Circadian-modulated activation
    """
    current_time = gmtime()
    hour = int(strftime("%H", current_time))
    minute = int(strftime("%M", current_time))
    
    # Convert to decimal hours
    decimal_hour = hour + minute / 60
    
    # Circadian rhythm (peaks around 2pm, troughs around 3am)
    circadian_factor = np.sin(2 * np.pi * (decimal_hour - 6) / 24)
    
    # Apply circadian modulation with ReLU-like base
    base_activation = F.relu(input)
    return base_activation * (1 + 0.2 * circadian_factor)


def wave_activation(input: torch.Tensor, frequency: float = 1.0, amplitude: float = 1.0) -> torch.Tensor:
    """
    Composite waveform activation.
    
    Args:
        input: Input tensor
        frequency: Wave frequency
        amplitude: Wave amplitude
        
    Returns:
        Wave-activated tensor
    """
    # Composite of sine and cosine waves
    sine_component = torch.sin(frequency * input)
    cosine_component = torch.cos(frequency * input)
    
    # Weighted combination
    return amplitude * (0.7 * sine_component + 0.3 * cosine_component)


def fourier_activation(input: torch.Tensor, weights: torch.Tensor, n_harmonics: int = 3) -> torch.Tensor:
    """
    Fourier series-based activation.
    
    Args:
        input: Input tensor
        weights: Harmonic weights tensor
        n_harmonics: Number of harmonics
        
    Returns:
        Fourier-activated tensor
    """
    result = torch.zeros_like(input)
    
    for i in range(n_harmonics):
        harmonic = (i + 1) * input
        weight = weights[i] if weights is not None else 1.0
        result += weight * torch.sin(harmonic) / (i + 1)
        
    return result


def quantum_activation(input: torch.Tensor, state_amplitudes: torch.Tensor, n_states: int = 2) -> torch.Tensor:
    """
    Quantum-inspired activation with superposition.
    
    Args:
        input: Input tensor
        state_amplitudes: Quantum state amplitudes
        n_states: Number of quantum states
        
    Returns:
        Quantum-activated tensor
    """
    # Normalize amplitudes (quantum normalization)
    normalized_amplitudes = F.softmax(state_amplitudes, dim=0)
    
    # Superposition of states
    result = torch.zeros_like(input)
    for i in range(n_states):
        phase = 2 * np.pi * i / n_states
        state_contribution = normalized_amplitudes[i] * torch.sin(input + phase)
        result += state_contribution
        
    return result


def lorenz_activation(input: torch.Tensor, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> torch.Tensor:
    """
    Lorenz attractor-inspired activation.
    
    Args:
        input: Input tensor
        sigma: Lorenz parameter σ
        rho: Lorenz parameter ρ
        beta: Lorenz parameter β
        
    Returns:
        Lorenz-activated tensor
    """
    # Simplified Lorenz dynamics applied to activation
    x = input
    y = torch.tanh(input)  # Bounded transformation
    z = torch.sigmoid(input)  # Another bounded transformation
    
    # One step of Lorenz dynamics
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    # Combine derivatives as activation
    return 0.1 * (dx + dy + dz)


def logistic_map_activation(input: torch.Tensor, r: float = 3.9, iterations: int = 3) -> torch.Tensor:
    """
    Logistic map chaos activation.
    
    Args:
        input: Input tensor
        r: Logistic map parameter
        iterations: Number of iterations
        
    Returns:
        Chaos-activated tensor
    """
    # Normalize inputs to [0, 1] range
    x = torch.sigmoid(input)
    
    # Apply logistic map iterations
    for _ in range(iterations):
        x = r * x * (1 - x)
        
    # Scale back to reasonable range
    return 2 * x - 1


def mandelbrot_activation(input: torch.Tensor, max_iterations: int = 5) -> torch.Tensor:
    """
    Mandelbrot set-inspired activation.
    
    Args:
        input: Input tensor
        max_iterations: Maximum iterations
        
    Returns:
        Mandelbrot-activated tensor
    """
    # Use inputs as complex numbers (real part only)
    z = torch.complex(input, torch.zeros_like(input))
    c = torch.complex(input * 0.5, torch.zeros_like(input))  # Scale down c
    
    # Mandelbrot iteration
    for _ in range(max_iterations):
        z = z * z + c
        
    # Return magnitude, bounded
    magnitude = torch.abs(z)
    return torch.tanh(magnitude)