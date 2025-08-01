#* Variables
SHELL ?= /usr/bin/env bash

# use the directory rather than the python binary to allow auto-discovery, which is more cross-platform compatible
PYTHON_PATH := $(shell which python)
# handle whether running on Windows or Unix-like systems
ifneq ($(findstring $(PYTHON_PATH),bin/python),)
	PYTHON_ROOT := $(shell dirname $(dir $(PYTHON_PATH)))
else
	PYTHON_ROOT := $(shell dirname $(PYTHON_PATH))
endif
ifeq ($(patsubst %/bin,,$(lastword $(PYTHON_ROOT))),)
  PYTHON_ROOT := $(dir $(PYTHON_ROOT))
endif
UV_PYTHON_ROOT ?= $(PYTHON_ROOT)

# to actually reuse an existing virtual/conda environment, the 'UV_PROJECT_ENVIRONMENT' variable must be set to it
# use this command:
#	UV_PROJECT_ENVIRONMENT=/path/to/env make [target]
# consider exporting this variable in '/path/to/env/etc/conda/activate.d/env.sh' to enable it by default when
# activating a conda environment, and reset it in '/path/to/env/etc/conda/deactivate.d/env.sh'
UV_PROJECT_ENVIRONMENT ?=
# make sure every uv command employs the specified environment path
ifeq (${UV_PROJECT_ENVIRONMENT},)
  UV_COMMAND := uv
else
  UV_COMMAND := UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" uv
endif

#* UV
.PHONY: setup
setup:
	which uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)

.PHONY: publish
publish:
	$(UV_COMMAND) publish --build

#* Installation
.PHONY: install
install: setup
	$(UV_COMMAND) export --format requirements-txt -o requirements.txt --no-dev
	$(UV_COMMAND) pip install --python "$(UV_PYTHON_ROOT)" -r requirements.txt

.PHONY: install-dev
install-dev: setup
	$(UV_COMMAND) export --format requirements-txt -o requirements-dev.txt
	$(UV_COMMAND) pip install --python "$(UV_PYTHON_ROOT)" -r requirements-dev.txt

.PHONY: install-dev-extras
install-dev-extras: setup
	$(UV_COMMAND) export --format requirements-txt -o requirements-dev.txt --extra torch
	$(UV_COMMAND) pip install --python "$(UV_PYTHON_ROOT)" -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/test/cpu --index-strategy unsafe-best-match

.PHONY: pre-commit-install
pre-commit-install: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" pre-commit install

#* Formatters
.PHONY: codestyle
codestyle: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" ruff format --config=pyproject.toml stac_model tests

.PHONY: format
format: codestyle

#* Testing
.PHONY: test
test: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" pytest -m "not slow" -c pyproject.toml -v --cov-report=html --cov=stac_model tests/

.PHONY: test-all
test-all: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" pytest -c pyproject.toml -v --cov-report=html --cov=stac_model tests/

#* Linting
.PHONY: check
check: check-examples check-markdown check-lint check-mypy check-safety check-citation

.PHONY: check-all
check-all: check

.PHONY: mypy
mypy: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" mypy --config-file pyproject.toml ./

.PHONY: check-mypy
check-mypy: mypy

.PHONY: check-safety
check-safety: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" safety check --full-report
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" bandit -ll --recursive stac_model tests

.PHONY: lint
lint: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" --extra torch ruff check --fix --config=pyproject.toml ./

.PHONY: check-lint
check-lint: lint
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" ruff check --config=pyproject.toml ./

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
update-dev-deps: setup
	$(UV_COMMAND) export --only-dev --format requirements-txt -o requirements-only-dev.txt
	$(UV_COMMAND) pip install --python "$(UV_PYTHON_ROOT)" -r requirements-only-dev.txt

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
