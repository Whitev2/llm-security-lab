.PHONY: install test lint run-harness

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests examples

run-harness:
	python -m examples.run_harness
