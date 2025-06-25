# CLAUDE.md - Development Guide for analogtivation

This document contains development guidelines, project structure, and implementation details for the analogtivation project overhaul.

## Project Overview

analogtivation is being transformed from a single experimental activation function into a comprehensive library of creative activation functions for deep learning, supporting multiple frameworks in both Python and R.

## Development Roadmap

### Phase 1: Foundation (Current)
1. **Analyze and document existing code**
   - Clock-based activation function in Jupyter notebook
   - MNIST experiment with time-based training
   - R package structure basics

2. **Design modular architecture**
   - Separate core math functions from framework implementations
   - Create consistent API across frameworks
   - Plan extensibility for new activation functions

### Phase 2: Core Implementation
3. **Python implementations**
   - Pure Python/NumPy base implementations
   - TensorFlow/Keras custom layers
   - PyTorch modules
   - JAX functions

4. **R implementations**
   - Base R functions
   - keras for R integration
   - torch for R modules

### Phase 3: Quality & Testing
5. **Testing strategy**
   - Unit tests for mathematical correctness
   - Integration tests with small models
   - Performance benchmarks
   - Cross-framework consistency tests

6. **CI/CD Pipeline**
   - GitHub Actions for automated testing
   - Multi-OS testing (Linux, macOS, Windows)
   - Python 3.7+ and R 4.0+ support

### Phase 4: Documentation & Release
7. **Documentation**
   - API reference
   - Tutorial notebooks
   - Mathematical explanations
   - Performance characteristics

8. **Release preparation**
   - Version 2.0.0 milestone
   - Migration guide from v1
   - PyPI and CRAN submission

## Technical Architecture

### Directory Structure
```
analogtivation/
├── python/
│   ├── analogtivation/
│   │   ├── __init__.py
│   │   ├── core/           # Pure implementations
│   │   ├── tensorflow/     # TF/Keras layers
│   │   ├── torch/          # PyTorch modules
│   │   └── jax/            # JAX functions
│   ├── tests/
│   ├── benchmarks/
│   └── setup.py
├── R/
│   ├── R/                  # R functions
│   ├── src/                # C++ implementations if needed
│   ├── tests/
│   └── vignettes/
├── examples/
│   ├── notebooks/
│   └── scripts/
└── docs/
```

### Activation Function Categories

1. **Time-Based**
   - Clock activation (original)
   - Seasonal activation
   - Circadian rhythm activation

2. **Wave-Based**
   - Composite waveforms
   - Fourier-based
   - Quantum-inspired

3. **Chaos-Based**
   - Lorenz attractor
   - Mandelbrot set
   - Logistic map

4. **Adaptive**
   - Learnable activations
   - Meta-learning activations

## Implementation Guidelines

### Python Code Style
- Follow PEP 8
- Type hints for all public functions
- Docstrings in NumPy style
- Black formatter for consistency

### R Code Style
- Follow tidyverse style guide
- roxygen2 documentation
- Use S3 methods for extensibility

### Testing Requirements
- Minimum 90% code coverage
- Test numerical stability
- Test edge cases (NaN, Inf, large values)
- Cross-framework consistency within 1e-6 tolerance

### Performance Considerations
- Vectorized implementations
- GPU compatibility
- Memory efficiency for large tensors
- Benchmark against standard activations

## Commands and Scripts

### Python Development
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/run_benchmarks.py

# Format code
black analogtivation/

# Type checking
mypy analogtivation/
```

### R Development
```r
# Install development dependencies
devtools::install_deps()

# Run tests
devtools::test()

# Check package
devtools::check()

# Build documentation
devtools::document()
```

### Release Process
```bash
# Version bump
# Update version in setup.py, DESCRIPTION, __init__.py

# Create changelog
# Update CHANGELOG.md with all changes

# Tag release
git tag -a v2.0.0 -m "Release version 2.0.0"
git push origin v2.0.0

# Python release
python setup.py sdist bdist_wheel
twine upload dist/*

# R release
devtools::release()
```

## Current Implementation Status

### Completed
- [x] Project planning and roadmap
- [x] README.md overhaul
- [x] Development documentation (this file)

### In Progress
- [ ] Analyzing existing clock activation implementation
- [ ] Designing modular architecture

### Pending
- [ ] All implementation tasks
- [ ] Testing framework setup
- [ ] CI/CD configuration
- [ ] Additional activation functions
- [ ] Performance optimization
- [ ] Documentation and examples
- [ ] Release preparation

## Notes for Future Development

1. **Reproducibility**: All stochastic activations should accept optional seeds
2. **Visualization**: Each activation should have a plot method
3. **Gradients**: Ensure all activations have well-defined gradients
4. **Numerical Stability**: Use log-sum-exp tricks where applicable
5. **Framework Compatibility**: Test with latest versions of all frameworks

## Contact

Project maintainer: Ryan San Juan (riosanjuan314@gmail.com)
Repository: https://github.com/rio-sanjuan/analogtivation