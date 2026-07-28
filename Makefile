.PHONY: install install-dev test lint format typecheck clean

# ── Installation ────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,examples]"

# ── Testing ──────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=ph_neuro --cov-report=term-missing

test-quick:
	pytest tests/ -v --tb=short -m "not slow"

# ── Linting & Formatting ──────────────────────────────────────────

lint:
	ruff check src/

format:
	ruff format src/ tests/

check-format:
	ruff format --check src/ tests/

# ── Type Checking ────────────────────────────────────────────────

typecheck:
	mypy src/ph_neuro/

# ── Cleanup ──────────────────────────────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# ── All checks (for CI / pre-commit) ─────────────────────────────

check: lint check-format typecheck test
