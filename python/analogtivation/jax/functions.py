"""JAX functional implementations of creative activation functions."""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from math import radians, tan
from time import gmtime, strftime
from typing import Optional, Tuple
from functools import partial


@partial(jit, static_argnums=(1,))
def clock_activation(x: jnp.ndarray, time_params: Optional[Tuple[float, float]] = None) -> jnp.ndarray:
    """
    Clock activation function that changes behavior based on current time.
    
    Note: Since JAX requires pure functions, time parameters must be passed in.
    If None, they will be computed outside JIT compilation.
    
    Args:
        x: Input array
        time_params: Optional tuple of (hour_slope, minute_slope)
        
    Returns:
        Activated array based on clock time
    """
    if time_params is None:
        # Compute time parameters outside JIT
        current_time = gmtime()
        hour = int(strftime("%H", current_time))
        minute = int(strftime("%M", current_time))
        second = int(strftime("%S", current_time))
        
        exact_hour = hour % 12 + minute / 60 + second / (60 * 60)
        exact_minute = minute + second / 60
        
        hour_hand_angle = -1 * (360 * exact_hour / 12 - 90)
        minute_hand_angle = -1 * (360 * exact_minute / 60 - 90)
        
        hour_slope = tan(radians(hour_hand_angle))
        minute_slope = tan(radians(minute_hand_angle))
    else:
        hour_slope, minute_slope = time_params
    
    # Apply different slopes based on sign
    return jnp.where(x >= 0, minute_slope * x, hour_slope * x)


@partial(jit, static_argnums=(1,))
def seasonal_activation(x: jnp.ndarray, seasonal_factor: Optional[float] = None) -> jnp.ndarray:
    """
    Activation function that varies with seasons.
    
    Args:
        x: Input array
        seasonal_factor: Optional pre-computed seasonal factor
        
    Returns:
        Seasonally adjusted activation
    """
    if seasonal_factor is None:
        # Compute seasonal factor outside JIT
        current_time = gmtime()
        day_of_year = int(strftime("%j", current_time))
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Apply seasonal modulation
    return x * (1 + 0.3 * seasonal_factor)


@partial(jit, static_argnums=(1,))
def circadian_activation(x: jnp.ndarray, circadian_factor: Optional[float] = None) -> jnp.ndarray:
    """
    24-hour circadian rhythm activation function.
    
    Args:
        x: Input array
        circadian_factor: Optional pre-computed circadian factor
        
    Returns:
        Circadian-modulated activation
    """
    if circadian_factor is None:
        # Compute circadian factor outside JIT
        current_time = gmtime()
        hour = int(strftime("%H", current_time))
        minute = int(strftime("%M", current_time))
        decimal_hour = hour + minute / 60
        circadian_factor = np.sin(2 * np.pi * (decimal_hour - 6) / 24)
    
    # Apply circadian modulation with ReLU-like base
    base_activation = jax.nn.relu(x)
    return base_activation * (1 + 0.2 * circadian_factor)


@jit
def wave_activation(x: jnp.ndarray, frequency: float = 1.0, amplitude: float = 1.0) -> jnp.ndarray:
    """
    Composite waveform activation.
    
    Args:
        x: Input array
        frequency: Wave frequency
        amplitude: Wave amplitude
        
    Returns:
        Wave-activated array
    """
    # Composite of sine and cosine waves
    sine_component = jnp.sin(frequency * x)
    cosine_component = jnp.cos(frequency * x)
    
    # Weighted combination
    return amplitude * (0.7 * sine_component + 0.3 * cosine_component)


@partial(jit, static_argnums=(2,))
def fourier_activation(x: jnp.ndarray, weights: jnp.ndarray, n_harmonics: int = 3) -> jnp.ndarray:
    """
    Fourier series-based activation.
    
    Args:
        x: Input array
        weights: Harmonic weights array
        n_harmonics: Number of harmonics
        
    Returns:
        Fourier-activated array
    """
    result = jnp.zeros_like(x)
    
    for i in range(n_harmonics):
        harmonic = (i + 1) * x
        weight = weights[i] if weights is not None else 1.0
        result += weight * jnp.sin(harmonic) / (i + 1)
        
    return result


@partial(jit, static_argnums=(2,))
def quantum_activation(x: jnp.ndarray, state_amplitudes: jnp.ndarray, n_states: int = 2) -> jnp.ndarray:
    """
    Quantum-inspired activation with superposition.
    
    Args:
        x: Input array
        state_amplitudes: Quantum state amplitudes
        n_states: Number of quantum states
        
    Returns:
        Quantum-activated array
    """
    # Normalize amplitudes (quantum normalization)
    normalized_amplitudes = jax.nn.softmax(state_amplitudes)
    
    # Superposition of states
    result = jnp.zeros_like(x)
    for i in range(n_states):
        phase = 2 * jnp.pi * i / n_states
        state_contribution = normalized_amplitudes[i] * jnp.sin(x + phase)
        result += state_contribution
        
    return result


@jit
def lorenz_activation(x: jnp.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3) -> jnp.ndarray:
    """
    Lorenz attractor-inspired activation.
    
    Args:
        x: Input array
        sigma: Lorenz parameter σ
        rho: Lorenz parameter ρ
        beta: Lorenz parameter β
        
    Returns:
        Lorenz-activated array
    """
    # Simplified Lorenz dynamics applied to activation
    y = jnp.tanh(x)  # Bounded transformation
    z = jax.nn.sigmoid(x)  # Another bounded transformation
    
    # One step of Lorenz dynamics
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    # Combine derivatives as activation
    return 0.1 * (dx + dy + dz)


@partial(jit, static_argnums=(1, 2))
def logistic_map_activation(x: jnp.ndarray, r: float = 3.9, iterations: int = 3) -> jnp.ndarray:
    """
    Logistic map chaos activation.
    
    Args:
        x: Input array
        r: Logistic map parameter
        iterations: Number of iterations
        
    Returns:
        Chaos-activated array
    """
    # Normalize inputs to [0, 1] range
    x_norm = jax.nn.sigmoid(x)
    
    # Apply logistic map iterations
    def logistic_step(carry, _):
        return r * carry * (1 - carry), None
    
    x_final, _ = jax.lax.scan(logistic_step, x_norm, None, length=iterations)
    
    # Scale back to reasonable range
    return 2 * x_final - 1


@partial(jit, static_argnums=(1,))
def mandelbrot_activation(x: jnp.ndarray, max_iterations: int = 5) -> jnp.ndarray:
    """
    Mandelbrot set-inspired activation.
    
    Args:
        x: Input array
        max_iterations: Maximum iterations
        
    Returns:
        Mandelbrot-activated array
    """
    # Use inputs as complex numbers (real part only)
    z = x + 0j
    c = x * 0.5 + 0j  # Scale down c
    
    # Mandelbrot iteration using scan for efficiency
    def mandelbrot_step(z_val, _):
        z_new = z_val * z_val + c
        return z_new, None
    
    z_final, _ = jax.lax.scan(mandelbrot_step, z, None, length=max_iterations)
    
    # Return magnitude, bounded
    magnitude = jnp.abs(z_final)
    return jnp.tanh(magnitude)