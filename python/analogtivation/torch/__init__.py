"""PyTorch implementations of creative activation functions."""

from .modules import (
    ClockActivation,
    SeasonalActivation,
    CircadianActivation,
    WaveActivation,
    FourierActivation,
    QuantumActivation,
    LorenzActivation,
    LogisticMapActivation,
    MandelbrotActivation,
)

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

__all__ = [
    # Modules
    "ClockActivation",
    "SeasonalActivation",
    "CircadianActivation",
    "WaveActivation",
    "FourierActivation",
    "QuantumActivation",
    "LorenzActivation",
    "LogisticMapActivation",
    "MandelbrotActivation",
    # Functional
    "clock_activation",
    "seasonal_activation",
    "circadian_activation",
    "wave_activation",
    "fourier_activation",
    "quantum_activation",
    "lorenz_activation",
    "logistic_map_activation",
    "mandelbrot_activation",
]