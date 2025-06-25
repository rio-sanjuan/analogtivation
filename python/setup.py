"""Setup configuration for analogtivation Python package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README file
here = Path(__file__).parent.absolute()
readme_path = here.parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8")

# Read version from __init__.py
version = "2.0.0"  # This should be kept in sync with __init__.py

setup(
    name="analogtivation",
    version=version,
    author="Ryan San Juan",
    author_email="riosanjuan314@gmail.com",
    description="Creative activation functions for deep learning inspired by "
    "real-world phenomena",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rio-sanjuan/analogtivation",
    project_urls={
        "Bug Tracker": "https://github.com/rio-sanjuan/analogtivation/issues",
        "Documentation": "https://analogtivation.readthedocs.io",
        "Source Code": "https://github.com/rio-sanjuan/analogtivation",
    },
    packages=find_packages(exclude=["tests", "tests.*", "benchmarks", "benchmarks.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.6.0"],
        "torch": ["torch>=1.9.0"],
        "jax": ["jax>=0.3.0", "jaxlib>=0.3.0"],
        "all": ["tensorflow>=2.6.0", "torch>=1.9.0", "jax>=0.3.0", "jaxlib>=0.3.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "hypothesis>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
            "nbsphinx>=0.9.0",
            "sphinx-copybutton>=0.5.0",
        ],
    },
    keywords=[
        "deep learning",
        "activation functions",
        "neural networks",
        "machine learning",
        "tensorflow",
        "pytorch",
        "jax",
    ],
    include_package_data=True,
    zip_safe=False,
)
