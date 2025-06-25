# Changelog

All notable changes to analogtivation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - Upcoming

### Added
- Complete project overhaul with modular architecture
- Python package with support for TensorFlow, PyTorch, and JAX
- R package with support for keras and torch
- New activation functions:
  - Seasonal activation (varies with time of year)
  - Circadian activation (24-hour cycle)
  - Wave-based activations
  - Chaos-based activations
  - Adaptive/learning activations
- Comprehensive documentation and examples
- Unit tests for all implementations
- CI/CD pipeline with GitHub Actions
- Performance benchmarks

### Changed
- Restructured project into separate Python and R packages
- Improved clock activation implementation
- Enhanced visualization capabilities
- Updated to modern package standards

### Fixed
- Numerical stability issues in original implementation
- Time zone handling

## [1.0.0] - 2021

### Added
- Initial implementation of clock activation function
- MNIST experiment with time-based training
- Basic R package structure
- Jupyter notebook with original concept