.PHONY: help install lint format typecheck test test-unit run clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install project dependencies with uv"
	@echo "  make lint       - Run ruff linter checks"
	@echo "  make format     - Run ruff code formatter"
	@echo "  make typecheck  - Run mypy static type analysis"
	@echo "  make test       - Run all pytest test suites"
	@echo "  make test-unit  - Run unit tests only (exclude integration)"
	@echo "  make run        - Launch Streamlit dashboard"
	@echo "  make clean      - Clean temporary files and caches"

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/

test:
	uv run pytest -v tests/

test-unit:
	uv run pytest -v -m "not integration" tests/

run:
	uv run streamlit run src/app.py

benchmark:
	uv run python scripts/run_benchmark.py

sample-data:
	uv run python scripts/download_sample_data.py --force-sample

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
