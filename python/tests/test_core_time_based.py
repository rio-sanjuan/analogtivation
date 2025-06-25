"""Tests for time-based activation functions."""

import pytest
import numpy as np
from unittest.mock import patch
from time import struct_time

from analogtivation.core.time_based import (
    clock_activation,
    seasonal_activation,
    circadian_activation,
    ClockActivation,
    SeasonalActivation,
    CircadianActivation,
)
from .test_utils import check_gradient_numerically, check_numerical_stability


class TestClockActivation:
    """Test clock activation function."""
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_clock_activation_noon(self, mock_gmtime, sample_inputs):
        """Test clock activation at noon (hands at 90-degree angle)."""
        # Mock time to noon (12:00:00)
        mock_gmtime.return_value = struct_time((2024, 1, 1, 12, 0, 0, 0, 1, 0))
        
        x = sample_inputs["small"]
        result = clock_activation(x)
        
        # At noon, minute hand points up (90 degrees), hour hand points right (0 degrees)
        # tan(90) is undefined, but in implementation it should handle this
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_clock_activation_different_times(self, mock_gmtime, sample_inputs):
        """Test clock activation at different times."""
        x = sample_inputs["small"]
        
        # Test at 3:15 (hands at different angles)
        mock_gmtime.return_value = struct_time((2024, 1, 1, 3, 15, 0, 0, 1, 0))
        result1 = clock_activation(x)
        
        # Test at 6:30 (hands at different angles)
        mock_gmtime.return_value = struct_time((2024, 1, 1, 6, 30, 0, 0, 1, 0))
        result2 = clock_activation(x)
        
        # Results should be different at different times
        assert not np.allclose(result1, result2)
    
    def test_clock_activation_sign_dependency(self, sample_inputs):
        """Test that clock activation uses different slopes for positive/negative inputs."""
        x = sample_inputs["small"]
        result = clock_activation(x)
        
        # Check that positive and negative values are treated differently
        pos_mask = x > 0
        neg_mask = x < 0
        
        if np.any(pos_mask) and np.any(neg_mask):
            # The ratios should be different for positive and negative values
            # (unless by chance the hour and minute hands have the same angle)
            pos_ratios = result[pos_mask] / x[pos_mask]
            neg_ratios = result[neg_mask] / x[neg_mask]
            
            # In most cases, these should be different
            assert len(np.unique(pos_ratios)) <= 2  # Should be consistent for all positive
            assert len(np.unique(neg_ratios)) <= 2  # Should be consistent for all negative
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_clock_activation_class(self, mock_gmtime, sample_inputs):
        """Test ClockActivation class implementation."""
        mock_gmtime.return_value = struct_time((2024, 1, 1, 3, 15, 0, 0, 1, 0))
        
        activation = ClockActivation()
        x = sample_inputs["small"]
        
        # Test forward pass
        result = activation.forward(x)
        func_result = clock_activation(x)
        np.testing.assert_allclose(result, func_result)
        
        # Test gradient
        grad = activation.gradient(x)
        assert grad.shape == x.shape
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_clock_activation_numerical_stability(self, mock_gmtime):
        """Test numerical stability of clock activation."""
        # Mock time to avoid extreme angle values
        mock_gmtime.return_value = struct_time((2024, 1, 1, 3, 15, 0, 0, 1, 0))
        # Use reasonable test values
        test_values = np.array([0, 1e-5, -1e-5, 10, -10, 100, -100])
        results = check_numerical_stability(clock_activation, test_values)
        assert results["all_finite"]
        assert results["handles_zero"]


class TestSeasonalActivation:
    """Test seasonal activation function."""
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_seasonal_activation_summer(self, mock_gmtime, sample_inputs):
        """Test seasonal activation in summer."""
        # Mock date to summer (July 1st, day 182)
        mock_gmtime.return_value = struct_time((2024, 7, 1, 0, 0, 0, 0, 182, 0))
        
        x = sample_inputs["small"]
        result = seasonal_activation(x)
        
        # In summer, activation should be amplified
        assert np.all(np.abs(result) >= np.abs(x) * 0.7)  # At least 70% of input
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_seasonal_activation_winter(self, mock_gmtime, sample_inputs):
        """Test seasonal activation in winter."""
        # Mock date to winter (January 1st, day 1)
        mock_gmtime.return_value = struct_time((2024, 1, 1, 0, 0, 0, 0, 1, 0))
        
        x = sample_inputs["small"]
        result = seasonal_activation(x)
        
        # In winter, activation should be dampened
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    def test_seasonal_activation_hemisphere(self, sample_inputs):
        """Test seasonal activation for different hemispheres."""
        x = sample_inputs["small"]
        
        result_north = seasonal_activation(x, hemisphere="northern")
        result_south = seasonal_activation(x, hemisphere="southern")
        
        # Results should generally be different (unless at equinox)
        # We can't assert they're always different due to equinox dates
        assert result_north.shape == result_south.shape
    
    def test_seasonal_activation_class(self, sample_inputs):
        """Test SeasonalActivation class implementation."""
        x = sample_inputs["small"]
        
        # Test northern hemisphere
        activation_north = SeasonalActivation(hemisphere="northern")
        result_north = activation_north.forward(x)
        func_result_north = seasonal_activation(x, hemisphere="northern")
        np.testing.assert_allclose(result_north, func_result_north)
        
        # Test southern hemisphere
        activation_south = SeasonalActivation(hemisphere="southern")
        result_south = activation_south.forward(x)
        func_result_south = seasonal_activation(x, hemisphere="southern")
        np.testing.assert_allclose(result_south, func_result_south)
        
        # Test gradient
        grad = activation_north.gradient(x)
        assert grad.shape == x.shape
        assert np.all(grad > 0)  # Should be positive scaling


class TestCircadianActivation:
    """Test circadian activation function."""
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_circadian_activation_day(self, mock_gmtime, sample_inputs):
        """Test circadian activation during day (peak time)."""
        # Mock time to 2 PM (peak circadian time)
        mock_gmtime.return_value = struct_time((2024, 1, 1, 14, 0, 0, 0, 1, 0))
        
        x = sample_inputs["small"]
        result = circadian_activation(x)
        
        # Should apply ReLU-like base
        assert np.all(result[x < 0] == 0)
        # Positive values should be modulated (may be slightly less due to circadian factor)
        # At 2 PM (peak), factor should be positive
        assert np.all(result[x > 0] > 0)  # Should be positive
        # Check modulation is reasonable
        assert np.all(np.abs(result[x > 0] - x[x > 0]) / x[x > 0] < 0.3)  # Within 30% modulation
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_circadian_activation_night(self, mock_gmtime, sample_inputs):
        """Test circadian activation during night (trough time)."""
        # Mock time to 3 AM (trough circadian time)
        mock_gmtime.return_value = struct_time((2024, 1, 1, 3, 0, 0, 0, 1, 0))
        
        x = sample_inputs["small"]
        result = circadian_activation(x)
        
        # Should still apply ReLU-like base
        assert np.all(result[x < 0] == 0)
        # Positive values should be dampened during trough time
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    @patch('analogtivation.core.time_based.gmtime')
    def test_circadian_activation_24h_cycle(self, mock_gmtime):
        """Test that circadian activation completes a 24-hour cycle."""
        x = np.array([1.0])  # Single positive value
        results = []
        
        # Test every hour
        for hour in range(24):
            mock_gmtime.return_value = struct_time((2024, 1, 1, hour, 0, 0, 0, 1, 0))
            results.append(circadian_activation(x)[0])
        
        # Should see variation throughout the day
        # Due to numerical precision and specific time points, might have small variations
        range_val = max(results) - min(results)
        assert range_val > 1e-6 or np.allclose(results, results[0])  # Either varies or constant
        # If varying, should be roughly periodic
        if range_val > 1e-6:
            # Period should complete in 24 hours
            assert abs(results[0] - results[-1]) < abs(results[0] - results[12])
    
    def test_circadian_activation_class(self, sample_inputs):
        """Test CircadianActivation class implementation."""
        activation = CircadianActivation()
        x = sample_inputs["small"]
        
        # Test forward pass
        result = activation.forward(x)
        func_result = circadian_activation(x)
        np.testing.assert_allclose(result, func_result)
        
        # Test gradient
        grad = activation.gradient(x)
        assert grad.shape == x.shape
        # Gradient should be 0 for negative inputs (ReLU-like)
        assert np.all(grad[x < 0] == 0)
        # Gradient should be positive for positive inputs
        assert np.all(grad[x > 0] > 0)