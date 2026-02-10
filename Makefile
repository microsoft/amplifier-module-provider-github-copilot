# =============================================================================
# Makefile for amplifier-module-provider-github-copilot
# =============================================================================
# Standard targets for development, testing, and quality assurance.
#
# Usage:
#   make install    - Install dev dependencies
#   make test       - Run all tests
#   make coverage   - Run tests with coverage report
#   make lint       - Check code style
#   make format     - Auto-format code
#   make check      - Run all checks (lint + test)
#   make clean      - Remove build artifacts
#
# =============================================================================

.PHONY: install test coverage lint format check clean help sdk-assumptions

# Default Python - override with: make test PYTHON=python3.12
PYTHON ?= python

# Package name for coverage
PACKAGE = amplifier_module_provider_github_copilot

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

install:
	$(PYTHON) -m pip install -e ".[dev]"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

# Run all tests (uses the 3-command pattern to avoid Windows asyncio issues)
test:
	@echo "Running tests (step 1/3: model naming)..."
	$(PYTHON) -m pytest tests/test_model_naming.py -q --tb=short
	@echo "Running tests (step 2/3: sync-heavy + SDK assumptions)..."
	$(PYTHON) -m pytest tests/test_models.py tests/test_converters.py tests/test_client.py \
		tests/test_exceptions.py tests/test_mount.py tests/test_mount_coverage.py \
		tests/test_sdk_driver.py tests/test_tool_capture.py tests/test_coverage_gaps.py \
		tests/sdk_assumptions/ -q --tb=short
	@echo "Running tests (step 3/3: async-heavy)..."
	$(PYTHON) -m pytest tests/test_provider.py tests/test_streaming.py -q --tb=short
	@echo "All tests passed!"

# Run SDK assumption tests only (use when upgrading SDK)
sdk-assumptions:
	$(PYTHON) -m pytest tests/sdk_assumptions/ -v --tb=long

# Run tests with coverage
coverage:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-report=term-missing --cov-report=html tests/
	@echo "Coverage report generated in htmlcov/index.html"

# Run tests with coverage and fail if under threshold
coverage-check:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-fail-under=80 tests/

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------

# Check code style without modifying
lint:
	ruff check $(PACKAGE)/ tests/

# Auto-format code
format:
	ruff check --fix --unsafe-fixes $(PACKAGE)/ tests/
	ruff format $(PACKAGE)/ tests/

# Run all checks (lint + test) - use before committing
check: lint test
	@echo "All checks passed!"

# Full pre-commit check including coverage
check-full: lint coverage-check
	@echo "Full checks passed!"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help:
	@echo "Available targets:"
	@echo "  install         - Install dev dependencies"
	@echo "  test            - Run all tests"
	@echo "  sdk-assumptions - Run SDK assumption tests only"
	@echo "  coverage        - Run tests with coverage report"
	@echo "  coverage-check  - Run tests with coverage threshold enforcement"
	@echo "  lint            - Check code style"
	@echo "  format          - Auto-format code"
	@echo "  check           - Run lint + test"
	@echo "  check-full      - Run lint + coverage with threshold"
	@echo "  clean           - Remove build artifacts"
