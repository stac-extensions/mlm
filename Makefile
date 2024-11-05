#* Variables
SHELL ?= /usr/bin/env bash
ACTIVEPYTHON = $(shell which python)

#* UV
.PHONY: setup
setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: publish
publish:
	uv publish --build

#* Installation
.PHONY: install
install:
	uv export --format requirements-txt -o requirements.txt --no-dev
	uv pip install --python $(ACTIVEPYTHON) -r requirements.txt

.PHONY: install-dev
install-dev:
	uv export --format requirements-txt -o requirements-dev.txt
	uv pip install --python $(ACTIVEPYTHON) -r requirements-dev.txt

.PHONY: pre-commit-install
pre-commit-install:
	uv run --python $(ACTIVEPYTHON) pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	uv run --python $(ACTIVEPYTHON) ruff format --config=pyproject.toml stac_model tests

.PHONY: format
format: codestyle

#* Linting
.PHONY: test
test:
	uv run --python $(ACTIVEPYTHON) pytest -c pyproject.toml --cov-report=html --cov=stac_model tests/

.PHONY: check
check: check-examples check-markdown check-lint check-mypy check-safety check-citation

.PHONY: check-all
check-all: check

.PHONY: mypy
mypy:
	uv run --python $(ACTIVEPYTHON) mypy --config-file pyproject.toml ./

.PHONY: check-mypy
check-mypy: mypy

.PHONY: check-safety
check-safety:
	uv run --python $(ACTIVEPYTHON) safety check --full-report
	uv run --python $(ACTIVEPYTHON) bandit -ll --recursive stac_model tests

.PHONY: lint
lint:
	uv run --python $(ACTIVEPYTHON) ruff check --fix --config=pyproject.toml ./

.PHONY: check-lint
check-lint: lint
	uv run --python $(ACTIVEPYTHON) ruff check --config=pyproject.toml ./

.PHONY: format-lint
format-lint: lint
	ruff format --config=pyproject.toml ./

.PHONY: install-npm
install-npm:
	npm install

.PHONY: check-markdown
check-markdown: install-npm
	npm run check-markdown

.PHONY: format-markdown
format-markdown: install-npm
	npm run format-markdown

.PHONY: check-examples
check-examples: install-npm
	npm run check-examples

.PHONY: format-examples
format-examples: install-npm
	npm run format-examples

FORMATTERS := lint markdown examples
$(addprefix fix-, $(FORMATTERS)): fix-%: format-%

.PHONY: lint-all
lint-all: lint mypy check-safety check-markdown

.PHONY: update-dev-deps
update-dev-deps:
	uv export --only-dev --format requirements-txt -o requirements-only-dev.txt
	uv pip install --python $(ACTIVEPYTHON) -r requirements-only-dev.txt

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
