.DEFAULT_GOAL := help

.PHONY: help install check lint format type-check test test-xml build clean-build publish

help: ## Display available commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install dependencies and pre-commit hooks
	uv sync --all-extras --dev
	uv run pre-commit install

# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

lint: ## Run ruff linter (with auto-fix)
	uv run ruff check . --fix

format: ## Run ruff formatter
	uv run ruff format .

type-check: ## Run ty type checker
	uv run ty check src/

check: ## Run all quality checks (lint, format check, type check)
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check src/

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: ## Run tests with terminal coverage report
	uv run pytest

test-xml: ## Run tests and produce XML coverage report only
	uv run pytest --cov-report=xml --cov-report=

# ---------------------------------------------------------------------------
# Build & publish
# ---------------------------------------------------------------------------

build: clean-build ## Build distribution packages
	uv build

clean-build: ## Remove build artefacts
	rm -rf dist/ build/ *.egg-info

publish: build ## Publish package to PyPI
	uv publish
