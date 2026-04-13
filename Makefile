# =============================================================================
# Makefile for amplifier-module-provider-github-copilot
# =============================================================================
# Standard targets for development, testing, and quality assurance.
#
# Usage:
#   make install    - Install dev dependencies
#   make test       - Run all tests
#   make smoke      - Quick E2E smoke test (seconds, not minutes)
#   make coverage   - Run tests with coverage report
#   make lint       - Check code style
#   make format     - Auto-format code
#   make check      - Run all checks (lint + test)
#   make clean      - Remove build artifacts
#
# =============================================================================

.PHONY: install test live smoke coverage coverage-check lint format check check-full clean help sdk-assumptions

# Default Python - override with: make test PYTHON=python3.12
# Use python3 for Linux/macOS compatibility (Debian/Ubuntu lack 'python' symlink)
PYTHON ?= python3

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

# Run all tests (simplified - let pytest handle collection)
test:
	@echo "Running all tests..."
	$(PYTHON) -m pytest tests/ -q --tb=short -m "not live"
	@echo "All tests passed!"

# Run SDK assumption tests only (use when upgrading SDK)
sdk-assumptions:
	$(PYTHON) -m pytest tests/test_sdk_assumptions.py -v --tb=long

# Run live integration tests (requires GITHUB_TOKEN, makes real API calls)
# Schedule: nightly only. Not for PR runs.
live:
	$(PYTHON) -m pytest tests/ -m live -v --tb=short

# Quick smoke test - validates provider works E2E in seconds
# Use after code changes, SDK upgrades, or to debug cross-platform issues
smoke:
	$(PYTHON) scripts/smoke_test.py --verbose

# Run tests with coverage
coverage:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-branch --cov-report=term-missing --cov-report=html tests/ -m "not live"
	@echo "Coverage report generated in htmlcov/index.html"

# Run tests with coverage and fail if under threshold
coverage-check:
	$(PYTHON) -m pytest --cov=$(PACKAGE) --cov-branch --cov-fail-under=90 tests/ -m "not live"

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
	@echo "  test            - Run all unit tests (excludes live)"
	@echo "  live            - Run live integration tests (requires GITHUB_TOKEN)"
	@echo "  sdk-assumptions - Run SDK assumption tests only (use when upgrading SDK)"
	@echo "  coverage        - Run unit tests with branch coverage report"
	@echo "  coverage-check  - Run unit tests with 90% threshold enforcement"
	@echo "  lint            - Check code style"
	@echo "  format          - Auto-format code"
	@echo "  check           - Run lint + unit tests"
	@echo "  check-full      - Run lint + coverage with threshold"
	@echo "  smoke           - Quick E2E smoke test"
	@echo "  clean           - Remove build artifacts"
