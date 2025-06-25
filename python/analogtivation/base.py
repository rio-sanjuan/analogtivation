"""Base classes and interfaces for activation functions."""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional
import numpy as np


class ActivationFunction(ABC):
    """Abstract base class for all activation functions."""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize activation function.
        
        Parameters
        ----------
        name : str
            Name of the activation function
        **kwargs
            Additional parameters specific to the activation
        """
        self.name = name
        self.params = kwargs
        
    @abstractmethod
    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Apply activation function.
        
        Parameters
        ----------
        x : array_like
            Input values
            
        Returns
        -------
        array_like
            Activated values
        """
        pass
    
    @abstractmethod
    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """
        Compute gradient of activation function.
        
        Parameters
        ----------
        x : array_like
            Input values
            
        Returns
        -------
        array_like
            Gradient values
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            'name': self.name,
            'params': self.params
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ActivationFunction':
        """Create activation from configuration."""
        return cls(**config.get('params', {}))
    
    def __call__(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Make activation callable."""
        return self.forward(x)


class TimeBasedActivation(ActivationFunction):
    """Base class for time-based activation functions."""
    
    def __init__(self, name: str, use_local_time: bool = True, **kwargs):
        """
        Initialize time-based activation.
        
        Parameters
        ----------
        name : str
            Name of the activation function
        use_local_time : bool
            Whether to use local time or UTC
        **kwargs
            Additional parameters
        """
        super().__init__(name, use_local_time=use_local_time, **kwargs)
        self.use_local_time = use_local_time
    
    @abstractmethod
    def get_time_factor(self) -> float:
        """Get current time-based factor."""
        pass


class AdaptiveActivation(ActivationFunction):
    """Base class for adaptive/learnable activation functions."""
    
    def __init__(self, name: str, learning_rate: float = 0.01, **kwargs):
        """
        Initialize adaptive activation.
        
        Parameters
        ----------
        name : str
            Name of the activation function
        learning_rate : float
            Learning rate for adaptation
        **kwargs
            Additional parameters
        """
        super().__init__(name, learning_rate=learning_rate, **kwargs)
        self.learning_rate = learning_rate
        self.state = {}
    
    @abstractmethod
    def update(self, x: np.ndarray, grad: np.ndarray) -> None:
        """
        Update activation parameters based on gradients.
        
        Parameters
        ----------
        x : np.ndarray
            Input values
        grad : np.ndarray
            Gradients from loss
        """
        pass