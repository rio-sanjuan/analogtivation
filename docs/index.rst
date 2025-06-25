analogtivation Documentation
============================

.. image:: https://img.shields.io/pypi/v/analogtivation.svg
   :target: https://pypi.org/project/analogtivation/
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/rio-sanjuan/analogtivation.svg
   :target: https://github.com/rio-sanjuan/analogtivation/blob/main/LICENSE
   :alt: License

**analogtivation** is a library of creative activation functions for deep learning, inspired by natural phenomena, physics, and mathematics. It provides novel activation functions that go beyond traditional ReLU, sigmoid, and tanh, offering researchers and practitioners new tools for neural network architectures.

Key Features
------------

* **Time-Based Activations**: Functions that change behavior based on time (clock, seasonal, circadian)
* **Wave-Based Activations**: Composite waveforms, Fourier series, and quantum-inspired functions
* **Chaos-Based Activations**: Lorenz attractor, Mandelbrot set, and logistic map implementations
* **Multi-Framework Support**: Works with TensorFlow/Keras, PyTorch, JAX, and R
* **Hardware Optimized**: GPU-accelerated implementations for all frameworks
* **Extensible Design**: Easy to add new activation functions

Installation
------------

Python
~~~~~~

.. code-block:: bash

   pip install analogtivation

For specific framework support:

.. code-block:: bash

   pip install analogtivation[tensorflow]  # For TensorFlow/Keras
   pip install analogtivation[torch]       # For PyTorch
   pip install analogtivation[jax]         # For JAX
   pip install analogtivation[all]         # For all frameworks

R
~

.. code-block:: r

   install.packages("analogtivation")
   # Or from GitHub:
   devtools::install_github("rio-sanjuan/analogtivation/R")

Quick Start
-----------

Python
~~~~~~

.. code-block:: python

   import numpy as np
   from analogtivation import clock_activation, wave_activation

   # Simple activation
   x = np.random.randn(100)
   y = clock_activation(x)

   # In TensorFlow/Keras
   from analogtivation.tensorflow import ClockActivation
   import tensorflow as tf

   model = tf.keras.Sequential([
       tf.keras.layers.Dense(64, input_shape=(10,)),
       ClockActivation(),
       tf.keras.layers.Dense(1)
   ])

   # In PyTorch
   from analogtivation.torch import WaveActivation
   import torch.nn as nn

   model = nn.Sequential(
       nn.Linear(10, 64),
       WaveActivation(frequency=2.0),
       nn.Linear(64, 1)
   )

R
~

.. code-block:: r

   library(analogtivation)

   # Basic usage
   x <- rnorm(100)
   y <- clock_activation(x)

   # With keras
   model <- keras_model_sequential() %>%
     layer_dense(units = 64, input_shape = 10) %>%
     layer_clock_activation() %>%
     layer_dense(units = 1)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   migration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/tensorflow
   api/torch
   api/jax
   api/r

.. toctree::
   :maxdepth: 2
   :caption: Mathematical Background

   math/time_based
   math/wave_based
   math/chaos_based
   math/adaptive

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   contributing
   architecture
   performance
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
