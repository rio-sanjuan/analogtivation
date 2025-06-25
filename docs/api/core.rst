Core API Reference
==================

This module contains the core activation functions implemented in pure Python/NumPy.

Time-Based Activations
----------------------

.. automodule:: analogtivation.core.time_based
   :members:
   :undoc-members:
   :show-inheritance:

Clock Activation
~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.clock_activation

The clock activation function modulates input based on the current time, creating temporal dynamics in neural networks.

**Mathematical Definition:**

.. math::

   f(x, t) = x \cdot \left( \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right) + \cos\left(\frac{2\pi \cdot \text{minute}}{60}\right) \right)

**Example:**

.. code-block:: python

   import numpy as np
   from analogtivation.core import clock_activation

   x = np.array([1.0, -0.5, 2.0])
   y = clock_activation(x)
   # Output varies based on current time

Seasonal Activation
~~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.seasonal_activation

Modulates activation based on day of year, useful for modeling seasonal patterns.

**Mathematical Definition:**

.. math::

   f(x, d) = x \cdot \left( 1 + \alpha \cdot \sin\left(\frac{2\pi \cdot d}{365.25}\right) \right)

where :math:`d` is the day of year and :math:`\alpha` is the seasonal strength.

Circadian Activation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.circadian_activation

Models biological circadian rhythms with customizable period and phase.

Wave-Based Activations
----------------------

.. automodule:: analogtivation.core.wave_based
   :members:
   :undoc-members:
   :show-inheritance:

Wave Activation
~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.wave_activation

Combines multiple sinusoidal waves with different frequencies and amplitudes.

**Mathematical Definition:**

.. math::

   f(x) = x + \sum_{i=1}^{n} A_i \cdot \sin(2\pi f_i \cdot x + \phi_i)

where :math:`A_i` are amplitudes, :math:`f_i` are frequencies, and :math:`\phi_i` are phases.

Fourier Activation
~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.fourier_activation

Uses Fourier series expansion for complex periodic patterns.

**Mathematical Definition:**

.. math::

   f(x) = a_0 + \sum_{n=1}^{N} \left[ a_n \cos(nx) + b_n \sin(nx) \right]

Quantum Activation
~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.quantum_activation

Quantum-inspired activation using superposition of states.

**Mathematical Definition:**

.. math::

   f(x) = \sum_{i=0}^{n-1} |i\rangle \langle i| \cdot P_i(x)

where :math:`P_i(x)` is the probability of state :math:`i` given input :math:`x`.

Chaos-Based Activations
-----------------------

.. automodule:: analogtivation.core.chaos_based
   :members:
   :undoc-members:
   :show-inheritance:

Lorenz Activation
~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.lorenz_activation

Based on the Lorenz attractor system from chaos theory.

**Mathematical Definition:**

The Lorenz system:

.. math::

   \frac{dx}{dt} &= \sigma(y - x) \\
   \frac{dy}{dt} &= x(\rho - z) - y \\
   \frac{dz}{dt} &= xy - \beta z

The activation uses numerical integration of this system.

Logistic Map Activation
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.logistic_map_activation

Iterates the logistic map equation, exhibiting chaotic behavior.

**Mathematical Definition:**

.. math::

   x_{n+1} = r \cdot x_n \cdot (1 - x_n)

where :math:`r` is the growth rate parameter.

Mandelbrot Activation
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.mandelbrot_activation

Inspired by the Mandelbrot set fractal.

**Mathematical Definition:**

.. math::

   z_{n+1} = z_n^2 + c

where :math:`c` is derived from the input and iteration count determines the output.

Adaptive Activations
--------------------

.. automodule:: analogtivation.core.adaptive
   :members:
   :undoc-members:
   :show-inheritance:

Learnable Activation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.learnable_activation

Parameterized activation function that can be learned during training.

Meta-Learning Activation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analogtivation.core.meta_activation

Activation that adapts based on task-specific metadata.

Utility Functions
-----------------

.. automodule:: analogtivation.core.utils
   :members:
   :undoc-members:
   :show-inheritance:
