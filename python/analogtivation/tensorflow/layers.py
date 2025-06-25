"""TensorFlow/Keras custom layer implementations of creative activation functions."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from math import radians, tan
from time import gmtime, strftime
from typing import Optional, Dict, Any


class ClockActivation(layers.Layer):
    """Clock activation layer that changes behavior based on current time."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        """Apply clock activation during forward pass."""
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
        positive_mask = tf.cast(inputs >= 0, inputs.dtype)
        negative_mask = tf.cast(inputs < 0, inputs.dtype)
        
        return (minute_slope * inputs * positive_mask + 
                hour_slope * inputs * negative_mask)
    
    def get_config(self):
        config = super().get_config()
        return config


class SeasonalActivation(layers.Layer):
    """Seasonal activation layer that varies with seasons."""
    
    def __init__(self, hemisphere: str = "northern", **kwargs):
        super().__init__(**kwargs)
        self.hemisphere = hemisphere
        
    def call(self, inputs):
        """Apply seasonal activation."""
        current_time = gmtime()
        day_of_year = int(strftime("%j", current_time))
        
        # Adjust for hemisphere
        if self.hemisphere == "southern":
            day_of_year = (day_of_year + 182) % 365
            
        # Calculate seasonal factor (peaks in summer, troughs in winter)
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Apply seasonal modulation
        return inputs * (1 + 0.3 * seasonal_factor)
    
    def get_config(self):
        config = super().get_config()
        config.update({"hemisphere": self.hemisphere})
        return config


class CircadianActivation(layers.Layer):
    """24-hour circadian rhythm activation layer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        """Apply circadian activation."""
        current_time = gmtime()
        hour = int(strftime("%H", current_time))
        minute = int(strftime("%M", current_time))
        
        # Convert to decimal hours
        decimal_hour = hour + minute / 60
        
        # Circadian rhythm (peaks around 2pm, troughs around 3am)
        circadian_factor = np.sin(2 * np.pi * (decimal_hour - 6) / 24)
        
        # Apply circadian modulation with ReLU-like base
        base_activation = tf.nn.relu(inputs)
        return base_activation * (1 + 0.2 * circadian_factor)
    
    def get_config(self):
        config = super().get_config()
        return config


class WaveActivation(layers.Layer):
    """Composite waveform activation layer."""
    
    def __init__(self, frequency: float = 1.0, amplitude: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.amplitude = amplitude
        
    def call(self, inputs):
        """Apply wave activation."""
        # Composite of sine and cosine waves
        sine_component = tf.sin(self.frequency * inputs)
        cosine_component = tf.cos(self.frequency * inputs)
        
        # Weighted combination
        return self.amplitude * (0.7 * sine_component + 0.3 * cosine_component)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "frequency": self.frequency,
            "amplitude": self.amplitude
        })
        return config


class FourierActivation(layers.Layer):
    """Fourier series-based activation layer."""
    
    def __init__(self, n_harmonics: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_harmonics = n_harmonics
        
    def build(self, input_shape):
        # Learnable weights for each harmonic
        self.harmonic_weights = self.add_weight(
            name="harmonic_weights",
            shape=(self.n_harmonics,),
            initializer="ones",
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        """Apply Fourier activation."""
        result = tf.zeros_like(inputs)
        
        for i in range(self.n_harmonics):
            harmonic = (i + 1) * inputs
            weight = self.harmonic_weights[i]
            result += weight * tf.sin(harmonic) / (i + 1)
            
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({"n_harmonics": self.n_harmonics})
        return config


class QuantumActivation(layers.Layer):
    """Quantum-inspired activation layer with superposition."""
    
    def __init__(self, n_states: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_states = n_states
        
    def build(self, input_shape):
        # Quantum state amplitudes
        self.state_amplitudes = self.add_weight(
            name="state_amplitudes",
            shape=(self.n_states,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        """Apply quantum activation."""
        # Normalize amplitudes (quantum normalization)
        normalized_amplitudes = tf.nn.softmax(self.state_amplitudes)
        
        # Superposition of states
        result = tf.zeros_like(inputs)
        for i in range(self.n_states):
            phase = 2 * np.pi * i / self.n_states
            state_contribution = normalized_amplitudes[i] * tf.sin(inputs + phase)
            result += state_contribution
            
        return result
    
    def get_config(self):
        config = super().get_config()
        config.update({"n_states": self.n_states})
        return config


class LorenzActivation(layers.Layer):
    """Lorenz attractor-inspired activation layer."""
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def call(self, inputs):
        """Apply Lorenz activation."""
        # Simplified Lorenz dynamics applied to activation
        # Using input as initial condition
        x = inputs
        y = tf.tanh(inputs)  # Bounded transformation
        z = tf.sigmoid(inputs)  # Another bounded transformation
        
        # One step of Lorenz dynamics
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        # Combine derivatives as activation
        return 0.1 * (dx + dy + dz)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma": self.sigma,
            "rho": self.rho,
            "beta": self.beta
        })
        return config


class LogisticMapActivation(layers.Layer):
    """Logistic map chaos activation layer."""
    
    def __init__(self, r: float = 3.9, iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.iterations = iterations
        
    def call(self, inputs):
        """Apply logistic map activation."""
        # Normalize inputs to [0, 1] range
        x = tf.nn.sigmoid(inputs)
        
        # Apply logistic map iterations
        for _ in range(self.iterations):
            x = self.r * x * (1 - x)
            
        # Scale back to reasonable range
        return 2 * x - 1
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "r": self.r,
            "iterations": self.iterations
        })
        return config


class MandelbrotActivation(layers.Layer):
    """Mandelbrot set-inspired activation layer."""
    
    def __init__(self, max_iterations: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        
    def call(self, inputs):
        """Apply Mandelbrot activation."""
        # Use inputs as complex numbers (real part only)
        z = tf.complex(inputs, tf.zeros_like(inputs))
        c = tf.complex(inputs * 0.5, tf.zeros_like(inputs))  # Scale down c
        
        # Mandelbrot iteration
        for _ in range(self.max_iterations):
            z = z * z + c
            
        # Return magnitude, bounded
        magnitude = tf.abs(z)
        return tf.tanh(magnitude)
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_iterations": self.max_iterations})
        return config