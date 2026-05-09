.PHONY: help dev test lint type bench fmt clean

PY := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy
PYTHONPATH := src

help:
	@echo "Targets:"
	@echo "  make test     Run unit + integration tests"
	@echo "  make bench    Run retrieval benchmark on the 10 reference traces"
	@echo "  make lint     Lint with ruff"
	@echo "  make type     Type-check with mypy --strict"
	@echo "  make fmt      Format with ruff"
	@echo "  make clean    Remove caches"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/

bench:
	PYTHONPATH=$(PYTHONPATH):. $(PY) -m eval.bench_retrieval

lint:
	$(RUFF) check src tests eval

type:
	$(MYPY)

fmt:
	$(RUFF) format src tests eval
	$(RUFF) check --fix src tests eval

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
