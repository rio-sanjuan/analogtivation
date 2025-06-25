#!/bin/bash
# Development environment setup script

echo "Setting up analogtivation development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install Python development dependencies
echo "Installing Python development dependencies..."
pip3 install --user black isort flake8 mypy pre-commit pytest pytest-cov

# Install the package in development mode
echo "Installing analogtivation in development mode..."
pip3 install --user -e python/[dev]

# Check if R is installed
if command -v R &> /dev/null; then
    echo "Installing R development dependencies..."
    Rscript -e "install.packages(c('devtools', 'testthat', 'styler', 'lintr'), repos='https://cran.rstudio.com/')"
else
    echo "R not found. Skipping R dependencies."
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo "Development environment setup complete!"
echo ""
echo "You can now use:"
echo "  make format     - Format all code"
echo "  make lint       - Run linting checks"
echo "  make test       - Run tests"
echo ""
echo "Pre-commit hooks will automatically format code on commit."