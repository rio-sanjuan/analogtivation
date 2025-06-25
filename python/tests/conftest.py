"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import os


@pytest.fixture
def sample_inputs():
    """Generate sample input arrays for testing."""
    return {
        "small": np.array([-2, -1, 0, 1, 2], dtype=np.float32),
        "medium": np.linspace(-5, 5, 100, dtype=np.float32),
        "large": np.random.randn(1000).astype(np.float32),
        "edge_cases": np.array([0, -0, np.inf, -np.inf, np.nan], dtype=np.float32),
    }


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons."""
    return 1e-6


@pytest.fixture
def frameworks_available():
    """Check which deep learning frameworks are available."""
    available = {}
    
    try:
        import tensorflow
        available["tensorflow"] = True
    except ImportError:
        available["tensorflow"] = False
        
    try:
        import torch
        available["torch"] = True
    except ImportError:
        available["torch"] = False
        
    try:
        import jax
        available["jax"] = True
    except ImportError:
        available["jax"] = False
        
    return available


@pytest.fixture
def skip_if_no_tensorflow(frameworks_available):
    """Skip test if TensorFlow is not available."""
    if not frameworks_available["tensorflow"]:
        pytest.skip("TensorFlow not available")


@pytest.fixture
def skip_if_no_torch(frameworks_available):
    """Skip test if PyTorch is not available."""
    if not frameworks_available["torch"]:
        pytest.skip("PyTorch not available")


@pytest.fixture
def skip_if_no_jax(frameworks_available):
    """Skip test if JAX is not available."""
    if not frameworks_available["jax"]:
        pytest.skip("JAX not available")


def assert_array_almost_equal(actual, expected, tolerance=1e-6):
    """Assert that two arrays are almost equal, handling NaN values."""
    # Check shapes match
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    
    # Handle NaN values
    nan_mask_actual = np.isnan(actual)
    nan_mask_expected = np.isnan(expected)
    assert np.array_equal(nan_mask_actual, nan_mask_expected), "NaN positions don't match"
    
    # Compare non-NaN values
    if not np.all(nan_mask_actual):
        np.testing.assert_allclose(
            actual[~nan_mask_actual],
            expected[~nan_mask_expected],
            rtol=tolerance,
            atol=tolerance
        )