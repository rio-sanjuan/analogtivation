"""Utility functions for testing activation functions."""

import numpy as np
from typing import Callable, Dict, Any, Optional
import warnings


def check_gradient_numerically(
    activation_func: Callable,
    gradient_func: Callable,
    x: np.ndarray,
    epsilon: float = 1e-7,
    tolerance: float = 1e-5
) -> bool:
    """
    Check if the gradient function matches numerical gradient.
    
    Parameters
    ----------
    activation_func : callable
        The activation function
    gradient_func : callable
        The gradient function to test
    x : np.ndarray
        Input values
    epsilon : float
        Small value for numerical differentiation
    tolerance : float
        Tolerance for comparison
        
    Returns
    -------
    bool
        True if gradients match within tolerance
    """
    # Compute analytical gradient
    analytical_grad = gradient_func(x)
    
    # Compute numerical gradient
    numerical_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        f_plus = activation_func(x_plus)[i]
        f_minus = activation_func(x_minus)[i]
        
        numerical_grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    # Compare gradients
    try:
        np.testing.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=tolerance,
            atol=tolerance
        )
        return True
    except AssertionError:
        return False


def check_activation_properties(
    activation_func: Callable,
    properties: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Check various mathematical properties of activation functions.
    
    Parameters
    ----------
    activation_func : callable
        The activation function to test
    properties : dict
        Dictionary of properties to check
        
    Returns
    -------
    dict
        Results of property checks
    """
    results = {}
    
    # Test data
    x = np.linspace(-5, 5, 100)
    
    # Check if function is bounded
    if "bounded" in properties:
        y = activation_func(x)
        is_bounded = np.all(np.isfinite(y))
        if properties["bounded"]:
            bounds = properties.get("bounds", (-np.inf, np.inf))
            is_bounded = is_bounded and np.all(y >= bounds[0]) and np.all(y <= bounds[1])
        results["bounded"] = is_bounded
    
    # Check if function is monotonic
    if "monotonic" in properties:
        y = activation_func(x)
        diffs = np.diff(y)
        if properties["monotonic"] == "increasing":
            results["monotonic"] = np.all(diffs >= -1e-10)
        elif properties["monotonic"] == "decreasing":
            results["monotonic"] = np.all(diffs <= 1e-10)
        else:
            results["monotonic"] = False
    
    # Check if function is odd/even
    if "symmetry" in properties:
        y_pos = activation_func(x[x > 0])
        y_neg = activation_func(-x[x > 0])
        
        if properties["symmetry"] == "odd":
            results["symmetry"] = np.allclose(y_pos, -y_neg, rtol=1e-6)
        elif properties["symmetry"] == "even":
            results["symmetry"] = np.allclose(y_pos, y_neg, rtol=1e-6)
    
    # Check zero-crossing
    if "zero_at_origin" in properties and properties["zero_at_origin"]:
        results["zero_at_origin"] = np.isclose(activation_func(0.0), 0.0, atol=1e-10)
    
    return results


def check_numerical_stability(
    activation_func: Callable,
    test_values: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Test numerical stability of activation function.
    
    Parameters
    ----------
    activation_func : callable
        The activation function to test
    test_values : np.ndarray, optional
        Specific test values, otherwise uses default edge cases
        
    Returns
    -------
    dict
        Results of stability tests
    """
    if test_values is None:
        test_values = np.array([
            0, -0,  # Zeros
            1e-10, -1e-10,  # Very small
            1e10, -1e10,  # Very large
            1e100, -1e100,  # Extremely large
            np.finfo(np.float32).eps,  # Machine epsilon
            -np.finfo(np.float32).eps,
        ])
    
    results = {
        "all_finite": True,
        "no_overflow": True,
        "no_underflow": True,
        "handles_zero": True,
        "issues": []
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        for val in test_values:
            try:
                result = activation_func(val)
                
                if not np.isfinite(result):
                    results["all_finite"] = False
                    results["issues"].append(f"Non-finite result for input {val}: {result}")
                
                if np.isinf(result):
                    results["no_overflow"] = False
                    
                if val != 0 and result == 0:
                    results["no_underflow"] = False
                    
            except Exception as e:
                results["issues"].append(f"Exception for input {val}: {str(e)}")
        
        # Check for warnings
        for warning in w:
            results["issues"].append(f"Warning: {warning.message}")
    
    return results