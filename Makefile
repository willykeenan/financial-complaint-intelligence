PYTHON ?= python3

.PHONY: help install-dev lint format-check test verify-evidence check

help:
	@printf '%s\n' \
		'install-dev  Install the package and development dependencies' \
		'lint         Run Ruff lint checks' \
		'format-check Verify Ruff formatting without changing files' \
		'test         Run the test suite' \
		'verify-evidence Validate published aggregate evidence' \
		'check        Run all static checks, tests, and evidence checks'

install-dev:
	$(PYTHON) -m pip install --editable '.[dev,demo]'

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

verify-evidence:
	$(PYTHON) scripts/verify_forward_holdout.py

check: lint format-check test verify-evidence
