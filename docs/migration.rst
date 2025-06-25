Migration Guide: v1.x to v2.0
==============================

This guide helps users migrate from analogtivation v1.x to v2.0. Version 2.0 is a complete rewrite with a new architecture, expanded functionality, and multi-framework support.

Breaking Changes
----------------

Package Structure
~~~~~~~~~~~~~~~~~

**v1.x:**

.. code-block:: python

   from analogtivation import AnalogActivation
   activation = AnalogActivation()

**v2.0:**

.. code-block:: python

   # Core functions
   from analogtivation.core import clock_activation

   # Framework-specific
   from analogtivation.tensorflow import ClockActivation
   from analogtivation.torch import ClockActivation as ClockActivationTorch

API Changes
~~~~~~~~~~~

1. **Function Names**: All activation functions now use snake_case naming:

   - ``AnalogActivation`` → ``clock_activation``
   - ``getActivation()`` → ``clock_activation()``

2. **Module Organization**: Functions are organized by type:

   - Time-based: ``analogtivation.core.time_based``
   - Wave-based: ``analogtivation.core.wave_based``
   - Chaos-based: ``analogtivation.core.chaos_based``

3. **Parameter Names**: More descriptive parameter names:

   - ``max_iterations`` → ``max_iter`` (Mandelbrot)
   - ``freq`` → ``frequency`` or ``frequencies``
   - ``amp`` → ``amplitude`` or ``amplitudes``

Feature Mapping
---------------

Clock Activation (formerly AnalogActivation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x:**

.. code-block:: python

   from analogtivation import AnalogActivation

   activation = AnalogActivation()
   output = activation.forward(input_tensor)

**v2.0 - NumPy:**

.. code-block:: python

   from analogtivation.core import clock_activation

   output = clock_activation(input_array)

**v2.0 - TensorFlow:**

.. code-block:: python

   from analogtivation.tensorflow import ClockActivation
   import tensorflow as tf

   model = tf.keras.Sequential([
       tf.keras.layers.Dense(64),
       ClockActivation(),
       tf.keras.layers.Dense(10)
   ])

**v2.0 - PyTorch:**

.. code-block:: python

   from analogtivation.torch import ClockActivation
   import torch.nn as nn

   model = nn.Sequential(
       nn.Linear(784, 64),
       ClockActivation(),
       nn.Linear(64, 10)
   )

New Features in v2.0
--------------------

1. **Multiple Activation Types**:

   - Wave-based activations
   - Fourier series activation
   - Quantum-inspired activation
   - Chaos-based activations (Lorenz, Mandelbrot, Logistic Map)
   - Learnable/adaptive activations

2. **Multi-Framework Support**:

   - TensorFlow/Keras
   - PyTorch
   - JAX
   - R (keras and torch)

3. **Performance Improvements**:

   - Vectorized operations
   - GPU acceleration
   - JIT compilation support (JAX)

4. **Better Testing**:

   - Comprehensive test suite
   - Property-based testing
   - Cross-framework consistency tests

Common Migration Patterns
-------------------------

Pattern 1: Simple Activation Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x:**

.. code-block:: python

   from analogtivation import AnalogActivation
   import numpy as np

   act = AnalogActivation()
   x = np.random.randn(100)
   y = act.forward(x)

**v2.0:**

.. code-block:: python

   from analogtivation.core import clock_activation
   import numpy as np

   x = np.random.randn(100)
   y = clock_activation(x)

Pattern 2: In Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x with Keras:**

.. code-block:: python

   from analogtivation import AnalogActivation
   from tensorflow.keras import layers, Model

   class CustomModel(Model):
       def __init__(self):
           super().__init__()
           self.dense1 = layers.Dense(64)
           self.activation = AnalogActivation()
           self.dense2 = layers.Dense(10)

       def call(self, x):
           x = self.dense1(x)
           x = self.activation.forward(x)
           return self.dense2(x)

**v2.0 with Keras:**

.. code-block:: python

   from analogtivation.tensorflow import ClockActivation
   import tensorflow as tf

   model = tf.keras.Sequential([
       tf.keras.layers.Dense(64),
       ClockActivation(),
       tf.keras.layers.Dense(10)
   ])

Pattern 3: Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**v1.x:**

.. code-block:: python

   act = AnalogActivation(scale=2.0, offset=0.5)

**v2.0:**

.. code-block:: python

   # Use wave activation for custom scaling
   from analogtivation.core import wave_activation

   y = wave_activation(x, frequencies=[1.0], amplitudes=[2.0])

Gradual Migration Strategy
--------------------------

1. **Install v2.0 alongside v1.x**:

   .. code-block:: bash

      pip install analogtivation==2.0.0

2. **Update imports gradually**:

   Start by updating imports in new code while keeping old code running.

3. **Test thoroughly**:

   The behavior of clock_activation in v2.0 should match AnalogActivation from v1.x, but verify with your specific use cases.

4. **Explore new features**:

   Once migration is complete, explore the new activation functions that might improve your models.

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you see:

.. code-block:: python

   ImportError: cannot import name 'AnalogActivation' from 'analogtivation'

Update to:

.. code-block:: python

   from analogtivation.core import clock_activation

Shape Mismatches
~~~~~~~~~~~~~~~~

v2.0 maintains input shapes for all activations. If you encounter shape issues, ensure you're not mixing framework-specific and NumPy implementations.

Performance Differences
~~~~~~~~~~~~~~~~~~~~~~~

v2.0 should be faster due to optimizations. If you notice slowdowns:

1. Ensure you're using the appropriate framework-specific implementation
2. Check that GPU acceleration is enabled (if applicable)
3. Review the batch sizes and data types

Getting Help
------------

- **Documentation**: https://analogtivation.readthedocs.io
- **GitHub Issues**: https://github.com/rio-sanjuan/analogtivation/issues
- **Discussions**: https://github.com/rio-sanjuan/analogtivation/discussions

For specific migration questions, please tag issues with `migration-v2`.
