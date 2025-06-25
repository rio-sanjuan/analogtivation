Installation Guide
==================

This guide covers installation of analogtivation for both Python and R environments.

Python Installation
-------------------

Requirements
~~~~~~~~~~~~

* Python 3.8 or higher
* NumPy 1.19.0 or higher
* (Optional) TensorFlow 2.6.0 or higher
* (Optional) PyTorch 1.9.0 or higher
* (Optional) JAX 0.3.0 or higher

Basic Installation
~~~~~~~~~~~~~~~~~~

Install the core package with all activation functions:

.. code-block:: bash

   pip install analogtivation

Framework-Specific Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with support for specific deep learning frameworks:

**TensorFlow/Keras:**

.. code-block:: bash

   pip install analogtivation[tensorflow]

**PyTorch:**

.. code-block:: bash

   pip install analogtivation[torch]

**JAX:**

.. code-block:: bash

   pip install analogtivation[jax]

**All Frameworks:**

.. code-block:: bash

   pip install analogtivation[all]

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or contributing:

.. code-block:: bash

   git clone https://github.com/rio-sanjuan/analogtivation.git
   cd analogtivation/python
   pip install -e ".[dev,all]"

This installs all frameworks plus development dependencies like pytest, black, and mypy.

R Installation
--------------

Requirements
~~~~~~~~~~~~

* R 4.1.0 or higher
* (Optional) keras for R
* (Optional) torch for R

CRAN Installation
~~~~~~~~~~~~~~~~~

.. code-block:: r

   install.packages("analogtivation")

GitHub Installation
~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: r

   # Install devtools if not already installed
   if (!requireNamespace("devtools", quietly = TRUE)) {
     install.packages("devtools")
   }

   # Install from GitHub
   devtools::install_github("rio-sanjuan/analogtivation/R")

With Framework Support
~~~~~~~~~~~~~~~~~~~~~~

For deep learning framework integration:

.. code-block:: r

   # For Keras support
   install.packages("keras")
   keras::install_keras()

   # For Torch support
   install.packages("torch")
   torch::install_torch()

Docker Installation
-------------------

We provide Docker images with all dependencies pre-installed:

.. code-block:: bash

   # CPU version
   docker pull ghcr.io/rio-sanjuan/analogtivation:latest

   # GPU version (with CUDA support)
   docker pull ghcr.io/rio-sanjuan/analogtivation:latest-gpu

Run with Docker:

.. code-block:: bash

   docker run -it --rm \
     -v $(pwd):/workspace \
     ghcr.io/rio-sanjuan/analogtivation:latest \
     python

Conda Installation
------------------

Using conda or mamba:

.. code-block:: bash

   # Create environment
   conda create -n analogtivation python=3.11
   conda activate analogtivation

   # Install package
   pip install analogtivation[all]

   # Or with conda-forge (when available)
   conda install -c conda-forge analogtivation

Verification
------------

Verify your installation:

**Python:**

.. code-block:: python

   import analogtivation
   print(analogtivation.__version__)

   # Test basic functionality
   import numpy as np
   x = np.random.randn(10)
   y = analogtivation.clock_activation(x)
   print(f"Input shape: {x.shape}, Output shape: {y.shape}")

**R:**

.. code-block:: r

   library(analogtivation)
   packageVersion("analogtivation")

   # Test basic functionality
   x <- rnorm(10)
   y <- clock_activation(x)
   cat("Input length:", length(x), "Output length:", length(y), "\n")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError with framework-specific modules:**

If you get an error like ``ImportError: No module named 'analogtivation.tensorflow'``, you need to install the framework-specific extras:

.. code-block:: bash

   pip install analogtivation[tensorflow]

**Version conflicts:**

If you encounter version conflicts, try creating a fresh virtual environment:

.. code-block:: bash

   python -m venv analogtivation-env
   source analogtivation-env/bin/activate  # On Windows: analogtivation-env\Scripts\activate
   pip install analogtivation[all]

**GPU support:**

For GPU acceleration with TensorFlow or PyTorch, ensure you have the appropriate CUDA drivers and framework-specific GPU packages installed.

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/rio-sanjuan/analogtivation/issues>`_
2. Review the `FAQ <faq.html>`_
3. Post a new issue with details about your environment and error messages
