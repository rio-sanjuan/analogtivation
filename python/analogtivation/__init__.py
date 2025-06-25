"""analogtivation: Creative activation functions for deep learning."""

__version__ = "2.0.0"
__author__ = "Ryan Johnson"
__email__ = "rtjohnson1206@gmail.com"

# Re-export base classes for extensibility
from .base import ActivationFunction, AdaptiveActivation, TimeBasedActivation

# Import all activation functions
from .core import *

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
