"""Time-based activation functions."""

from math import radians, tan
from time import gmtime, strftime
from typing import Union

import numpy as np

from ..base import TimeBasedActivation


def to_clock_angle(theta: float) -> float:
    """Convert angle to clock notation (12 o'clock = 90 degrees)."""
    return -1 * (theta - 90)


def clock_activation(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Clock activation function that changes behavior based on current time.

    Uses minute hand angle for positive inputs and hour hand angle for negative inputs.

    Parameters
    ----------
    x : array_like
        Input values

    Returns
    -------
    array_like
        Activated values based on current clock time
    """
    current_time = gmtime()

    hour = int(strftime("%H", current_time))
    minute = int(strftime("%M", current_time))
    second = int(strftime("%S", current_time))

    # Calculate exact positions
    exact_hour = hour % 12 + minute / 60 + second / (60 * 60)
    exact_minute = minute + second / 60

    # Convert to angles
    hour_hand_angle = to_clock_angle(360 * exact_hour / 12)
    minute_hand_angle = to_clock_angle(360 * exact_minute / 60)

    # Calculate slopes
    hour_slope = tan(radians(hour_hand_angle))
    minute_slope = tan(radians(minute_hand_angle))

    # Apply different slopes based on sign
    x_array = np.asarray(x)
    result = np.where(x_array >= 0, minute_slope * x_array, hour_slope * x_array)

    return result if isinstance(x, np.ndarray) else float(result)


def seasonal_activation(
    x: Union[np.ndarray, float], hemisphere: str = "northern"
) -> Union[np.ndarray, float]:
    """
    Activation function that varies with seasons.

    Parameters
    ----------
    x : array_like
        Input values
    hemisphere : str, optional
        Either "northern" or "southern" hemisphere

    Returns
    -------
    array_like
        Seasonally adjusted activation
    """
    current_time = gmtime()
    day_of_year = int(strftime("%j", current_time))

    # Adjust for hemisphere
    if hemisphere == "southern":
        day_of_year = (day_of_year + 182) % 365

    # Calculate seasonal factor (peaks in summer, troughs in winter)
    seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Apply seasonal modulation
    x_array = np.asarray(x)
    result = x_array * (1 + 0.3 * seasonal_factor)

    return result if isinstance(x, np.ndarray) else float(result)


def circadian_activation(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    24-hour circadian rhythm activation function.

    Parameters
    ----------
    x : array_like
        Input values

    Returns
    -------
    array_like
        Circadian-modulated activation
    """
    current_time = gmtime()
    hour = int(strftime("%H", current_time))
    minute = int(strftime("%M", current_time))

    # Convert to decimal hours
    decimal_hour = hour + minute / 60

    # Circadian rhythm (peaks around 2pm, troughs around 3am)
    circadian_factor = np.sin(2 * np.pi * (decimal_hour - 6) / 24)

    # Apply circadian modulation with ReLU-like base
    x_array = np.asarray(x)
    base_activation = np.maximum(0, x_array)
    result = base_activation * (1 + 0.2 * circadian_factor)

    return result if isinstance(x, np.ndarray) else float(result)


class ClockActivation(TimeBasedActivation):
    """Clock activation function class implementation."""

    def __init__(self, use_local_time: bool = True):
        """Initialize clock activation."""
        super().__init__("clock", use_local_time=use_local_time)

    def get_time_factor(self) -> tuple:
        """Get current time factors."""
        current_time = gmtime()

        hour = int(strftime("%H", current_time))
        minute = int(strftime("%M", current_time))
        second = int(strftime("%S", current_time))

        exact_hour = hour % 12 + minute / 60 + second / (60 * 60)
        exact_minute = minute + second / 60

        hour_angle = to_clock_angle(360 * exact_hour / 12)
        minute_angle = to_clock_angle(360 * exact_minute / 60)

        return tan(radians(hour_angle)), tan(radians(minute_angle))

    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply clock activation."""
        hour_slope, minute_slope = self.get_time_factor()
        x_array = np.asarray(x)
        result = np.where(x_array >= 0, minute_slope * x_array, hour_slope * x_array)
        return result if isinstance(x, np.ndarray) else float(result)

    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of clock activation."""
        hour_slope, minute_slope = self.get_time_factor()
        x_array = np.asarray(x)
        grad = np.where(x_array >= 0, minute_slope, hour_slope)
        return grad if isinstance(x, np.ndarray) else float(grad)


class SeasonalActivation(TimeBasedActivation):
    """Seasonal activation function class implementation."""

    def __init__(self, hemisphere: str = "northern", use_local_time: bool = True):
        """Initialize seasonal activation."""
        super().__init__(
            "seasonal", hemisphere=hemisphere, use_local_time=use_local_time
        )
        self.hemisphere = hemisphere

    def get_time_factor(self) -> float:
        """Get seasonal factor."""
        current_time = gmtime()
        day_of_year = int(strftime("%j", current_time))

        if self.hemisphere == "southern":
            day_of_year = (day_of_year + 182) % 365

        return np.sin(2 * np.pi * (day_of_year - 80) / 365)

    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply seasonal activation."""
        seasonal_factor = self.get_time_factor()
        x_array = np.asarray(x)
        result = x_array * (1 + 0.3 * seasonal_factor)
        return result if isinstance(x, np.ndarray) else float(result)

    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of seasonal activation."""
        seasonal_factor = self.get_time_factor()
        return np.ones_like(x) * (1 + 0.3 * seasonal_factor)


class CircadianActivation(TimeBasedActivation):
    """Circadian rhythm activation function class implementation."""

    def __init__(self, use_local_time: bool = True):
        """Initialize circadian activation."""
        super().__init__("circadian", use_local_time=use_local_time)

    def get_time_factor(self) -> float:
        """Get circadian factor."""
        current_time = gmtime()
        hour = int(strftime("%H", current_time))
        minute = int(strftime("%M", current_time))

        decimal_hour = hour + minute / 60
        return np.sin(2 * np.pi * (decimal_hour - 6) / 24)

    def forward(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Apply circadian activation."""
        circadian_factor = self.get_time_factor()
        x_array = np.asarray(x)
        base_activation = np.maximum(0, x_array)
        result = base_activation * (1 + 0.2 * circadian_factor)
        return result if isinstance(x, np.ndarray) else float(result)

    def gradient(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        """Compute gradient of circadian activation."""
        circadian_factor = self.get_time_factor()
        x_array = np.asarray(x)
        grad = np.where(x_array > 0, 1 + 0.2 * circadian_factor, 0)
        return grad if isinstance(x, np.ndarray) else float(grad)
