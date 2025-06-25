"""Setup configuration for analogtivation Python package."""

from setuptools import find_packages, setup

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="analogtivation",
    version="2.0.0",
    author="Ryan San Juan",
    author_email="riosanjuan314@gmail.com",
    description="Creative activation functions for deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rtjohnson12/analogtivation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0,<3.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.15.0,<3.0"],
        "torch": ["torch>=2.0.0,<3.0"],
        "jax": ["jax>=0.4.0,<1.0", "jaxlib>=0.4.0,<1.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "mypy>=0.910",
            "flake8>=3.9",
            "isort>=5.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
    },
)
