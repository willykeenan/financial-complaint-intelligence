PYTHON ?= python3

.PHONY: help install-dev lint format-check test check

help:
	@printf '%s\n' \
		'install-dev  Install the package and development dependencies' \
		'lint         Run Ruff lint checks' \
		'format-check Verify Ruff formatting without changing files' \
		'test         Run the test suite' \
		'check        Run all static checks and tests'

install-dev:
	$(PYTHON) -m pip install --editable '.[dev,demo]'

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

check: lint format-check test
