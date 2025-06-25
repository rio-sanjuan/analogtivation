"""analogtivation: Creative activation functions for deep learning."""

__version__ = "2.0.0"
__author__ = "Ryan Johnson"
__email__ = "riosanjuan314@gmail.com"

# Re-export base classes for extensibility
from .base import ActivationFunction, AdaptiveActivation, TimeBasedActivation

# Import all activation functions
from .core import (  # Time-based; Wave-based; Chaos-based
    CircadianActivation,
    ClockActivation,
    FourierActivation,
    LogisticMapActivation,
    LorenzActivation,
    MandelbrotActivation,
    QuantumActivation,
    SeasonalActivation,
    WaveActivation,
    circadian_activation,
    clock_activation,
    fourier_activation,
    logistic_map_activation,
    lorenz_activation,
    mandelbrot_activation,
    quantum_activation,
    seasonal_activation,
    wave_activation,
)

__all__ = [
    # Base classes
    "ActivationFunction",
    "TimeBasedActivation",
    "AdaptiveActivation",
    # Time-based activations
    "clock_activation",
    "seasonal_activation",
    "circadian_activation",
    "ClockActivation",
    "SeasonalActivation",
    "CircadianActivation",
    # Wave-based activations
    "wave_activation",
    "fourier_activation",
    "quantum_activation",
    "WaveActivation",
    "FourierActivation",
    "QuantumActivation",
    # Chaos-based activations
    "lorenz_activation",
    "logistic_map_activation",
    "mandelbrot_activation",
    "LorenzActivation",
    "LogisticMapActivation",
    "MandelbrotActivation",
]
