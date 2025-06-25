Mathematical Foundations: Time-Based Activations
================================================

This document provides the mathematical theory behind time-based activation functions.

Clock Activation
----------------

Overview
~~~~~~~~

The clock activation function uses the positions of clock hands to modulate neural network activations, creating a unique time-dependent transformation.

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

The clock activation function is defined as:

.. math::

   f(x, t) = \begin{cases}
   x \cdot \tan(\theta_m(t)) & \text{if } x \geq 0 \\
   x \cdot \tan(\theta_h(t)) & \text{if } x < 0
   \end{cases}

where:

- :math:`\theta_m(t)` is the minute hand angle at time :math:`t`
- :math:`\theta_h(t)` is the hour hand angle at time :math:`t`

Clock Hand Angles
~~~~~~~~~~~~~~~~~

The angles are calculated as:

.. math::

   \theta_h(t) = \frac{2\pi}{12} \left( h + \frac{m}{60} + \frac{s}{3600} \right) - \frac{\pi}{2}

   \theta_m(t) = \frac{2\pi}{60} \left( m + \frac{s}{60} \right) - \frac{\pi}{2}

where :math:`h`, :math:`m`, and :math:`s` are hours (12-hour format), minutes, and seconds respectively.

The :math:`-\frac{\pi}{2}` adjustment converts from standard mathematical angles (0° at 3 o'clock) to clock notation (0° at 12 o'clock).

Properties
~~~~~~~~~~

1. **Periodicity**: The function has different periods for positive and negative inputs:

   - Positive inputs: 60-minute period (minute hand)
   - Negative inputs: 12-hour period (hour hand)

2. **Discontinuities**: The tangent function creates discontinuities when clock hands point to 6 o'clock (straight down), where :math:`\tan(\theta) \to \infty`.

3. **Range**: Without bounds, the output range is :math:`(-\infty, \infty)`. In practice, outputs are typically clamped to :math:`[-10, 10]` for numerical stability.

4. **Gradient**:

.. math::

   \frac{\partial f}{\partial x} = \begin{cases}
   \tan(\theta_m(t)) & \text{if } x \geq 0 \\
   \tan(\theta_h(t)) & \text{if } x < 0
   \end{cases}

Time Dependency Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

The activation strength varies throughout the day:

- **Maximum positive activation**: When minute hand points to 3 o'clock (:math:`\theta_m = 0°`)
- **Maximum negative activation**: When minute hand points to 9 o'clock (:math:`\theta_m = 180°`)
- **Zero activation**: When hands point to 12 or 6 o'clock

Seasonal Activation
-------------------

Overview
~~~~~~~~

The seasonal activation function modulates inputs based on the day of year, creating annual cyclic patterns.

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   f(x, d) = x \cdot \left( 1 + \alpha \cdot \sin\left(\frac{2\pi(d - \phi)}{365.25}\right) \right)

where:

- :math:`d` is the day of year (1-365/366)
- :math:`\alpha` is the modulation strength (default: 0.3)
- :math:`\phi` is the phase offset (default: 80, aligning peak with summer solstice)

Hemisphere Adjustment
~~~~~~~~~~~~~~~~~~~~~

For the southern hemisphere, the day is shifted by 182 days:

.. math::

   d_{south} = (d_{north} + 182) \mod 365

Properties
~~~~~~~~~~

1. **Range**: For input :math:`x` and modulation :math:`\alpha = 0.3`:

   .. math::

      f(x, d) \in [0.7x, 1.3x]

2. **Extrema**:

   - Maximum: Around June 21 (day 172) for northern hemisphere
   - Minimum: Around December 21 (day 355) for northern hemisphere

3. **Gradient**:

.. math::

   \frac{\partial f}{\partial x} = 1 + \alpha \cdot \sin\left(\frac{2\pi(d - \phi)}{365.25}\right)

Circadian Activation
--------------------

Overview
~~~~~~~~

Models biological circadian rhythms that follow a 24-hour cycle, typically used to simulate alertness patterns.

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   f(x, t) = x \cdot A(t)

where :math:`A(t)` is the alertness function:

.. math::

   A(t) = \begin{cases}
   0.7 + 0.3 \cdot \sin\left(\frac{\pi(h - 6)}{6}\right) & \text{if } 6 \leq h < 12 \\
   1.0 & \text{if } 12 \leq h < 14 \\
   1.0 - 0.2 \cdot \sin\left(\frac{\pi(h - 14)}{4}\right) & \text{if } 14 \leq h < 18 \\
   0.8 - 0.3 \cdot \sin\left(\frac{\pi(h - 18)}{6}\right) & \text{if } 18 \leq h < 24 \\
   0.5 + 0.2 \cdot \sin\left(\frac{\pi h}{6}\right) & \text{if } 0 \leq h < 6
   \end{cases}

where :math:`h` is the hour of day (0-23).

Biological Basis
~~~~~~~~~~~~~~~~

The alertness function approximates human circadian patterns:

- **Early morning (0-6h)**: Low alertness, gradual awakening
- **Morning (6-12h)**: Rising alertness
- **Midday (12-14h)**: Peak alertness
- **Afternoon (14-18h)**: Post-lunch dip
- **Evening (18-24h)**: Declining alertness

Properties
~~~~~~~~~~

1. **Range**: :math:`A(t) \in [0.5, 1.0]`, so :math:`f(x, t) \in [0.5x, x]`

2. **Continuity**: The function is designed to be continuous at all transition points

3. **Biological accuracy**: Parameters are based on chronobiology research

Applications
------------

Time-based activations are particularly useful for:

1. **Time Series Prediction**: Incorporating known temporal patterns
2. **Adaptive Systems**: Networks that change behavior based on time
3. **Biological Modeling**: Simulating circadian or seasonal biological processes
4. **Financial Models**: Capturing trading hour or seasonal effects
5. **Energy Systems**: Modeling daily or seasonal demand patterns

Implementation Considerations
-----------------------------

1. **Time Zones**: Current implementations use UTC time. Consider time zone adjustments for local applications.

2. **Numerical Stability**: The tangent function in clock activation requires careful handling near discontinuities.

3. **Gradient Flow**: Time-based modulation can affect gradient flow during backpropagation.

4. **Reproducibility**: Results vary with time, making exact reproducibility challenging without time mocking.
