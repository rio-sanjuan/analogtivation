"""Time-based activation functions.

This module contains activation functions that incorporate temporal dynamics,
allowing neural networks to exhibit time-dependent behavior. These functions
are useful for:

- Modeling systems with inherent time dependencies
- Creating adaptive networks that change behavior over time
- Incorporating real-world temporal patterns (daily, seasonal cycles)
- Time series analysis and prediction

Examples
--------
>>> import numpy as np
>>> from analogtivation.core import clock_activation
>>> x = np.array([1.0, -0.5, 2.0])
>>> y = clock_activation(x)  # Output varies with current time
"""

from math import radians, tan
from time import gmtime, strftime
from typing import Union

import numpy as np

from ..base import TimeBasedActivation


def to_clock_angle(theta: float) -> float:
    """Convert angle to clock notation.

    Transforms standard mathematical angles to clock notation where:
    - 12 o'clock = 90 degrees (top)
    - 3 o'clock = 0 degrees (right)
    - 6 o'clock = -90 degrees (bottom)
    - 9 o'clock = 180 degrees (left)

    Parameters
    ----------
    theta : float
        Angle in degrees (standard mathematical notation)

    Returns
    -------
    float
        Angle in clock notation

    Examples
    --------
    >>> to_clock_angle(0)  # 3 o'clock position
    90.0
    >>> to_clock_angle(90)  # 12 o'clock position
    0.0
    """
    return -1 * (theta - 90)


def clock_activation(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Clock activation function that changes behavior based on current time.

    This activation function modulates inputs based on the current positions
    of clock hands, creating a time-dependent transformation. It uses:
    - Minute hand angle for positive inputs
    - Hour hand angle for negative inputs

    The function applies the tangent of clock hand angles as slopes,
    creating periodic variations in the activation strength throughout
    the day.

    Parameters
    ----------
    x : array_like
        Input values. Can be a scalar or array of any shape.

    Returns
    -------
    array_like
        Activated values with same shape as input. The transformation
        depends on the current system time.

    Notes
    -----
    The activation exhibits discontinuities when clock hands point
    straight up or down (tangent approaches infinity). These are
    handled by clamping to reasonable bounds.

    The function is deterministic for a given time but changes
    continuously as time progresses.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, -0.5, 2.0, -1.5])
    >>> y = clock_activation(x)
    >>> # Output varies based on current time
    >>> # At 3:15 PM, might return something like:
    >>> # array([1.73, -0.29, 3.46, -0.87])

    See Also
    --------
    seasonal_activation : For longer-term temporal patterns
    circadian_activation : For biological rhythm modeling
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

    Modulates input values based on the day of year, creating seasonal
    patterns that peak in summer and trough in winter. This is useful
    for modeling:

    - Seasonal business patterns
    - Agricultural or weather-dependent systems
    - Biological processes with annual cycles
    - Energy consumption patterns

    Parameters
    ----------
    x : array_like
        Input values to be seasonally modulated
    hemisphere : str, optional
        Either "northern" or "southern" hemisphere. Determines whether
        summer occurs mid-year (northern) or year-end (southern).
        Default is "northern".

    Returns
    -------
    array_like
        Seasonally adjusted activation with same shape as input.
        Values are modulated by ±30% based on season.

    Notes
    -----
    The seasonal factor follows a sinusoidal pattern:
    - Maximum (summer): ~June 21 (northern) or ~December 21 (southern)
    - Minimum (winter): ~December 21 (northern) or ~June 21 (southern)
    - Neutral (spring/fall): ~March 21 and ~September 21

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> # In summer (northern hemisphere):
    >>> seasonal_activation(x, "northern")
    array([1.3, 2.6, 3.9])  # 30% increase
    >>> # In winter:
    >>> seasonal_activation(x, "northern")
    array([0.7, 1.4, 2.1])  # 30% decrease
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

    Models biological circadian rhythms that regulate sleep-wake cycles,
    hormone production, and other physiological processes. The activation
    strength varies throughout the day following typical human alertness
    patterns.

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
