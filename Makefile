.PHONY: help format format-python format-r lint install-dev test clean

help:
	@echo "Available commands:"
	@echo "  make format        - Format all code (Python and R)"
	@echo "  make format-python - Format Python code only"
	@echo "  make format-r      - Format R code only"
	@echo "  make lint          - Run linting checks"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean build artifacts"

format: format-python format-r

format-python:
	@echo "Formatting Python code..."
	@black python/
	@isort python/

format-r:
	@echo "Formatting R code..."
	@Rscript -e "styler::style_pkg()"

lint:
	@echo "Running Python linting..."
	@flake8 python/ --max-line-length=88 --extend-ignore=E203,W503
	@mypy python/analogtivation/
	@echo "Running R linting..."
	@Rscript -e "lintr::lint_package()"

install-dev:
	@echo "Installing Python development dependencies..."
	@pip install black isort flake8 mypy pre-commit
	@pip install -e python/[dev]
	@echo "Installing R development dependencies..."
	@Rscript -e "install.packages(c('styler', 'lintr'), repos='https://cran.rstudio.com/')"
	@echo "Installing pre-commit hooks..."
	@pre-commit install

test:
	@echo "Running Python tests..."
	@cd python && python -m pytest tests/
	@echo "Running R tests..."
	@Rscript -e "devtools::test()"

clean:
	@echo "Cleaning build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf python/build python/dist python/*.egg-info
	@rm -rf .pytest_cache .mypy_cache