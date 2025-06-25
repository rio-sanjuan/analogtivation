"""Chaos-based activation functions."""

import numpy as np
from typing import Union
from ..base import ActivationFunction


class LorenzActivation(ActivationFunction):
    """
    Lorenz attractor-based activation function.
    
    Uses the chaotic dynamics of the Lorenz system.
    """
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        """
        Initialize Lorenz activation.
        
        Parameters
        ----------
        sigma : float
            Prandtl number
        rho : float
            Rayleigh number
        beta : float
            Geometric factor
        """
        super().__init__("lorenz", sigma=sigma, rho=rho, beta=beta)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        
    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply Lorenz activation."""
        x_array = np.asarray(x)
        
        # Initialize Lorenz system at x-dependent point
        state_x = x_array
        state_y = np.tanh(x_array)  # Bounded initialization
        state_z = np.abs(x_array)
        
        # Single iteration of Lorenz dynamics
        dx = self.sigma * (state_y - state_x)
        dy = state_x * (self.rho - state_z) - state_y
        dz = state_x * state_y - self.beta * state_z
        
        # Combine derivatives with tanh for bounded output
        result = np.tanh(dx + dy + dz)
        
        return result if isinstance(x, np.ndarray) else float(result)
    
    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of Lorenz activation."""
        epsilon = 1e-7
        x_array = np.asarray(x)
        
        # Numerical gradient due to chaotic nature
        grad = (self.forward(x_array + epsilon) - self.forward(x_array - epsilon)) / (2 * epsilon)
        
        return grad if isinstance(x, np.ndarray) else float(grad)


class LogisticMapActivation(ActivationFunction):
    """
    Logistic map chaos activation function.
    
    Uses the chaotic behavior of the logistic map.
    """
    
    def __init__(self, r: float = 3.9, iterations: int = 3):
        """
        Initialize logistic map activation.
        
        Parameters
        ----------
        r : float
            Growth rate parameter (chaos when r > 3.57)
        iterations : int
            Number of map iterations
        """
        super().__init__("logistic_map", r=r, iterations=iterations)
        self.r = r
        self.iterations = iterations
        
    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply logistic map activation."""
        x_array = np.asarray(x)
        
        # Map input to (0, 1) interval
        state = 1 / (1 + np.exp(-x_array))
        
        # Apply logistic map iterations
        for _ in range(self.iterations):
            state = self.r * state * (1 - state)
            
        # Scale output to reasonable range
        result = 2 * state - 1
        
        return result if isinstance(x, np.ndarray) else float(result)
    
    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of logistic map activation."""
        x_array = np.asarray(x)
        
        # Initial sigmoid and its derivative
        sigmoid = 1 / (1 + np.exp(-x_array))
        d_sigmoid = sigmoid * (1 - sigmoid)
        
        # Track gradient through iterations
        state = sigmoid
        grad = d_sigmoid
        
        for _ in range(self.iterations):
            # Gradient of logistic map: r * (1 - 2*state)
            grad = grad * self.r * (1 - 2 * state)
            state = self.r * state * (1 - state)
            
        # Scale gradient
        grad = 2 * grad
        
        return grad if isinstance(x, np.ndarray) else float(grad)


class MandelbrotActivation(ActivationFunction):
    """
    Mandelbrot set-inspired activation function.
    
    Uses fractal properties for activation.
    """
    
    def __init__(self, max_iter: int = 5, escape_radius: float = 2.0):
        """
        Initialize Mandelbrot activation.
        
        Parameters
        ----------
        max_iter : int
            Maximum iterations before escape
        escape_radius : float
            Radius for escape condition
        """
        super().__init__("mandelbrot", max_iter=max_iter, escape_radius=escape_radius)
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        
    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply Mandelbrot activation."""
        x_array = np.asarray(x)
        
        # Use input as complex parameter
        c = x_array + 0j
        z = 0 + 0j
        
        # Iterate Mandelbrot formula
        for i in range(self.max_iter):
            z = z**2 + c
            
            # Check escape condition
            if isinstance(x_array, np.ndarray):
                mask = np.abs(z) > self.escape_radius
                abs_z = np.abs(z)
                abs_z = np.where(abs_z == 0, 1, abs_z)  # Avoid division by zero
                z = np.where(mask, z / abs_z * self.escape_radius, z)
            else:
                if abs(z) > self.escape_radius:
                    z = z / abs(z) * self.escape_radius
        
        # Return real part normalized
        result = np.real(z) / self.escape_radius
        
        return result if isinstance(x, np.ndarray) else float(result)
    
    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of Mandelbrot activation."""
        epsilon = 1e-7
        x_array = np.asarray(x)
        
        # Numerical gradient for fractal function
        grad = (self.forward(x_array + epsilon) - self.forward(x_array - epsilon)) / (2 * epsilon)
        
        return grad if isinstance(x, np.ndarray) else float(grad)


# Convenience functions for functional API
def lorenz_activation(x: Union[np.ndarray, float],
                     sigma: float = 10.0,
                     rho: float = 28.0,
                     beta: float = 8/3) -> Union[np.ndarray, float]:
    """Functional interface for Lorenz activation."""
    return LorenzActivation(sigma, rho, beta).forward(x)


def logistic_map_activation(x: Union[np.ndarray, float],
                           r: float = 3.9,
                           iterations: int = 3) -> Union[np.ndarray, float]:
    """Functional interface for logistic map activation."""
    return LogisticMapActivation(r, iterations).forward(x)


def mandelbrot_activation(x: Union[np.ndarray, float],
                         max_iter: int = 5,
                         escape_radius: float = 2.0) -> Union[np.ndarray, float]:
    """Functional interface for Mandelbrot activation."""
    return MandelbrotActivation(max_iter, escape_radius).forward(x)