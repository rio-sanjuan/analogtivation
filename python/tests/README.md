# Test Suite for Analogtivation

This directory contains comprehensive tests for the analogtivation library.

## Test Structure

### Core Implementation Tests
- `test_core_time_based.py` - Tests for clock, seasonal, and circadian activations
- `test_core_wave_based.py` - Tests for wave, Fourier, and quantum activations  
- `test_core_chaos_based.py` - Tests for Lorenz, logistic map, and Mandelbrot activations

### Framework-Specific Tests
- `test_tensorflow_*.py` - Tests for TensorFlow/Keras implementations (to be added)
- `test_torch_*.py` - Tests for PyTorch implementations (to be added)
- `test_jax_*.py` - Tests for JAX implementations (to be added)

### Cross-Framework Tests
- `test_cross_framework.py` - Ensures consistency across all framework implementations

### Utilities
- `conftest.py` - Pytest configuration and shared fixtures
- `test_utils.py` - Helper functions for gradient checking, stability testing, etc.

## Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run only core tests
pytest tests/test_core_*.py

# Run with coverage
pytest --cov=analogtivation --cov-report=html

# Run specific test
pytest tests/test_core_time_based.py::TestClockActivation -v
```

## Test Categories

### 1. Functionality Tests
- Basic operation verification
- Parameter variation tests
- Edge case handling

### 2. Mathematical Property Tests
- Boundedness verification
- Symmetry checking
- Monotonicity validation

### 3. Numerical Stability Tests
- Large/small value handling
- NaN/Inf propagation
- Overflow/underflow prevention

### 4. Gradient Tests
- Numerical gradient verification
- Backpropagation consistency

### 5. Cross-Framework Consistency
- Output matching across frameworks
- Gradient consistency
- Time synchronization for time-based functions

## Known Issues and Fixes Applied

1. **Time-based mocking**: Fixed by patching the specific import path
2. **Parameter naming**: Updated to match actual implementation (e.g., `max_iter` vs `max_iterations`)
3. **Numerical stability**: Added bounded test values to prevent overflow
4. **Lorenz chaos**: Adjusted expectations due to tanh bounding

## Future Enhancements

1. Add performance benchmarks
2. Implement property-based testing with Hypothesis
3. Add visual tests for activation function shapes
4. Create integration tests with real neural networks