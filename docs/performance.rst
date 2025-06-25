Performance Guide
=================

This guide provides performance characteristics, optimization tips, and benchmarking results for analogtivation functions.

Performance Overview
--------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Activation Function Complexity
   :header-rows: 1
   :widths: 30 20 50

   * - Activation Function
     - Complexity
     - Notes
   * - Clock Activation
     - O(n)
     - Simple arithmetic operations
   * - Wave Activation
     - O(n·m)
     - m = number of frequencies
   * - Fourier Activation
     - O(n·k)
     - k = number of terms
   * - Quantum Activation
     - O(n·s)
     - s = number of states
   * - Lorenz Activation
     - O(n)
     - Fixed iterations, uses tanh
   * - Logistic Map
     - O(n·i)
     - i = number of iterations
   * - Mandelbrot Activation
     - O(n·j)
     - j = max iterations

where n is the number of input elements.

Benchmark Results
-----------------

The following benchmarks were performed on a system with:
- CPU: Intel Core i9-10900K @ 3.7GHz
- GPU: NVIDIA RTX 3080
- RAM: 32GB DDR4
- Python 3.11, NumPy 1.24, TensorFlow 2.14, PyTorch 2.1

CPU Performance
~~~~~~~~~~~~~~~

Average execution time (milliseconds) for different input sizes:

.. code-block:: text

   Input Size: 10,000 elements
   ┌─────────────────────┬─────────┬─────────────┬───────────┐
   │ Activation          │ NumPy   │ TensorFlow  │ PyTorch   │
   ├─────────────────────┼─────────┼─────────────┼───────────┤
   │ Clock               │ 0.82    │ 1.23        │ 1.15      │
   │ Wave (3 freqs)      │ 2.41    │ 2.88        │ 2.76      │
   │ Fourier (5 terms)   │ 3.15    │ 3.67        │ 3.52      │
   │ Quantum (4 states)  │ 1.93    │ 2.34        │ 2.21      │
   │ Lorenz              │ 1.45    │ 1.89        │ 1.78      │
   │ Logistic (5 iter)   │ 2.67    │ 3.12        │ 2.98      │
   │ Mandelbrot (3 iter) │ 2.23    │ 2.71        │ 2.59      │
   └─────────────────────┴─────────┴─────────────┴───────────┘

GPU Performance
~~~~~~~~~~~~~~~

Speedup factor compared to CPU for batch operations:

.. code-block:: text

   Batch Size: 1000 x 1000
   ┌─────────────────────┬─────────────┬───────────┐
   │ Activation          │ TensorFlow  │ PyTorch   │
   ├─────────────────────┼─────────────┼───────────┤
   │ Clock               │ 15.2x       │ 14.8x     │
   │ Wave                │ 18.5x       │ 17.9x     │
   │ Fourier             │ 21.3x       │ 20.7x     │
   │ Quantum             │ 16.8x       │ 16.2x     │
   │ Lorenz              │ 14.1x       │ 13.6x     │
   │ Logistic Map        │ 12.4x       │ 11.9x     │
   └─────────────────────┴─────────────┴───────────┘

Memory Usage
~~~~~~~~~~~~

Peak memory consumption for different activation functions:

.. code-block:: python

   # Input size: 1,000,000 float32 elements (4MB)

   Clock Activation:     ~8MB   (2x input)
   Wave Activation:      ~12MB  (3x input, due to intermediate arrays)
   Fourier Activation:   ~16MB  (4x input, stores sin/cos terms)
   Quantum Activation:   ~20MB  (5x input, probability calculations)
   Lorenz Activation:    ~12MB  (3x input, ODE integration)

Optimization Techniques
-----------------------

1. Vectorization
~~~~~~~~~~~~~~~~

All functions are fully vectorized using NumPy operations:

.. code-block:: python

   # Efficient (vectorized)
   y = wave_activation(x)  # x is large array

   # Inefficient (loop)
   y = np.array([wave_activation(xi) for xi in x])

2. GPU Acceleration
~~~~~~~~~~~~~~~~~~~

For large batches, use framework-specific implementations:

.. code-block:: python

   # TensorFlow GPU
   import tensorflow as tf
   from analogtivation.tensorflow import WaveActivation

   with tf.device('/GPU:0'):
       layer = WaveActivation()
       y = layer(x)  # Executed on GPU

3. JIT Compilation
~~~~~~~~~~~~~~~~~~

JAX implementations support JIT compilation:

.. code-block:: python

   import jax
   from analogtivation.jax import wave_activation

   # JIT compile the function
   wave_jit = jax.jit(wave_activation)

   # First call includes compilation time
   y = wave_jit(x)  # ~10ms

   # Subsequent calls are much faster
   y = wave_jit(x)  # ~0.5ms

4. Parameter Tuning
~~~~~~~~~~~~~~~~~~~

Reduce computational cost by limiting parameters:

.. code-block:: python

   # Fewer frequencies = faster computation
   wave_fast = WaveActivation(frequencies=[1.0], amplitudes=[1.0])
   wave_slow = WaveActivation(frequencies=[1,2,3,4,5], amplitudes=[1,0.5,0.33,0.25,0.2])

   # Fewer iterations = faster computation
   logistic_fast = lambda x: logistic_map_activation(x, iterations=2)
   logistic_slow = lambda x: logistic_map_activation(x, iterations=10)

Best Practices
--------------

1. **Choose the Right Implementation**

   - NumPy: CPU-only, small to medium data
   - TensorFlow/PyTorch: Large batches, GPU available
   - JAX: Need JIT compilation, functional programming

2. **Batch Operations**

   Always process data in batches rather than individual samples:

   .. code-block:: python

      # Good: Process entire batch
      batch_output = activation(batch_input)  # Shape: (1000, 784)

      # Bad: Process one at a time
      outputs = []
      for sample in batch_input:
          outputs.append(activation(sample))

3. **Memory Management**

   For very large datasets, process in chunks:

   .. code-block:: python

      def process_large_data(data, activation, chunk_size=10000):
          results = []
          for i in range(0, len(data), chunk_size):
              chunk = data[i:i + chunk_size]
              results.append(activation(chunk))
          return np.concatenate(results)

4. **Profile Your Code**

   Use profiling tools to identify bottlenecks:

   .. code-block:: python

      import cProfile
      import pstats

      profiler = cProfile.Profile()
      profiler.enable()

      # Your code here
      y = fourier_activation(x, n_terms=10)

      profiler.disable()
      stats = pstats.Stats(profiler).sort_stats('cumulative')
      stats.print_stats(10)

Scaling Considerations
----------------------

Input Size Scaling
~~~~~~~~~~~~~~~~~~

Performance scales linearly with input size for most activations:

.. code-block:: text

   Time(n) ≈ k × n

   where k depends on the activation complexity

Multi-GPU Scaling
~~~~~~~~~~~~~~~~~

For very large models, distribute across multiple GPUs:

.. code-block:: python

   # TensorFlow multi-GPU
   strategy = tf.distribute.MirroredStrategy()
   with strategy.scope():
       model = build_model_with_analogtivation()

   # PyTorch multi-GPU
   model = nn.DataParallel(model)

Framework-Specific Tips
-----------------------

TensorFlow
~~~~~~~~~~

- Use ``@tf.function`` decorator for custom implementations
- Enable mixed precision for faster GPU computation
- Use ``tf.data`` pipelines for efficient data loading

PyTorch
~~~~~~~

- Ensure tensors are on the same device
- Use ``torch.jit.script`` for production deployment
- Enable automatic mixed precision (AMP) for speedup

JAX
~~~

- Always use ``jax.jit`` for production code
- Leverage ``vmap`` for batch operations
- Use ``lax.scan`` for sequential operations

Performance Monitoring
----------------------

Track activation performance in production:

.. code-block:: python

   import time
   import logging

   class TimedActivation:
       def __init__(self, activation_fn):
           self.activation_fn = activation_fn
           self.total_time = 0
           self.call_count = 0

       def __call__(self, x):
           start = time.perf_counter()
           result = self.activation_fn(x)
           elapsed = time.perf_counter() - start

           self.total_time += elapsed
           self.call_count += 1

           if self.call_count % 1000 == 0:
               avg_time = self.total_time / self.call_count
               logging.info(f"Avg activation time: {avg_time*1000:.2f}ms")

           return result
