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
ifeq ($(UV_PROJECT_ENVIRONMENT),)
  UV_COMMAND := uv
  # auto-detect conda environment for consistency
  ifneq ($(CONDA_PREFIX),)
    ifeq ($(shell realpath $(UV_PYTHON_ROOT)),$(shell realpath $(CONDA_PREFIX)))
      UV_COMMAND := UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}" uv
    endif
  endif
else
  UV_COMMAND := UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" uv
endif

env:
	@echo "PYTHON_PATH: $(PYTHON_PATH)"
	@echo "PYTHON_ROOT: $(PYTHON_ROOT)"
	@echo "UV_PYTHON_ROOT: $(UV_PYTHON_ROOT)"
	@echo "UV_PROJECT_ENVIRONMENT: $(UV_PROJECT_ENVIRONMENT)"
	@echo "UV_COMMAND: $(UV_COMMAND)"


# auto-detect N_CUDA_DEVICES, or override explicitly during make command as desired
N_CUDA_DEVICES = $(shell which nvidia-smi >/dev/null && nvidia-smi -L | wc -l || echo '0')
TORCH_EXTRA = $(if $(filter-out $(N_CUDA_DEVICES),0),torch-cu126,torch)
.PHONY: cuda
cuda:
	@echo "Available CUDA devices: $(N_CUDA_DEVICES)"
	@echo "Would install using extra: $(TORCH_EXTRA)"

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
	$(UV_COMMAND) export --format requirements-txt -o requirements-dev.txt
	$(UV_COMMAND) pip install --python "$(UV_PYTHON_ROOT)" -e ".[$(TORCH_EXTRA)]" -r requirements-dev.txt

.PHONY: update
update: setup  # install package updates, optionally to a specific package or all
	$(UV_COMMAND) sync --python "$(UV_PYTHON_ROOT)" --resolution highest $(if $(PIP_PACKAGE),-P $(PIP_PACKAGE),-U) $(UV_XARGS)

.PHONY: update-dev
update-dev: setup  # install package updates with developement dependencies
	$(UV_COMMAND) sync --python "$(UV_PYTHON_ROOT)" --resolution highest --group dev $(UV_XARGS)

.PHONY: update-extras
update-extras: setup  # install package updates with extra torch dependencies
	$(UV_COMMAND) sync --python "$(UV_PYTHON_ROOT)" --resolution highest --extra $(TORCH_EXTRA) $(UV_XARGS)

.PHONY: update-all
update-all: setup  # install package updates with all dependencies
	$(UV_COMMAND) sync --python "$(UV_PYTHON_ROOT)" --resolution highest --all-groups --extra $(TORCH_EXTRA) $(UV_XARGS)

.PHONY: pre-commit-install
pre-commit-install: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" pre-commit install

#* Formatters
.PHONY: codestyle
codestyle: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" ruff format --config=pyproject.toml stac_model tests

.PHONY: format
format: codestyle

#* Testing
.PHONY: test
test: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" pytest -c pyproject.toml -v --cov-report=html --cov=stac_model --cov-config pyproject.toml tests/ $(PYTEST_XARGS)

#* Linting
.PHONY: check
check: check-examples check-markdown check-lint check-mypy check-safety check-citation

.PHONY: check-all
check-all: check

.PHONY: check-warn-torch
check-warn-torch:
	@$(UV_COMMAND) pip list --quiet --python "$(UV_PYTHON_ROOT)" | grep '^torch\s' >/dev/null || \
		(echo "Warning: 'torch' is not installed in the current environment. Following operations could report invalid check results." >&2)

.PHONY: check-warn-torchvision
check-warn-torchvision:
	@$(UV_COMMAND) pip list --quiet --python "$(UV_PYTHON_ROOT)" | grep '^torchvision\s' >/dev/null || \
		(echo "Warning: 'torchvision' is not installed in the current environment. Following operations could report invalid check results." >&2)

.PHONY: mypy
mypy: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" mypy --config-file pyproject.toml ./

.PHONY: check-mypy
check-mypy: check-warn-torch check-warn-torchvision mypy

.PHONY: check-safety
check-safety: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" safety check --full-report
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" bandit -ll --recursive stac_model tests

.PHONY: check-citation
check-citation: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" cffconvert --validate

# see https://docs.astral.sh/ruff/formatter/#sorting-imports for use of both `check` and `format` commands
.PHONY: lint
lint: setup
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" ruff check --select I --fix --config=pyproject.toml ./
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" ruff format --config=pyproject.toml ./

.PHONY: check-lint
check-lint: setup
	$(UV_COMMAND) run --python "$(UV_PYTHON_ROOT)" ruff check --config=pyproject.toml ./

.PHONY: format-lint
format-lint: lint
	$(UV_COMMAND) run --no-sync --python "$(UV_PYTHON_ROOT)" ruff check --fix --config=pyproject.toml ./

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

.PHONY: mypy-cache-remove
mypy-cache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: notebooks-cache-remove
notebooks-cache-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytest-cache-remove
pytest-cache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: cache-remove
cache-remove: pycache-remove dsstore-remove myp-ycache-remove notebooks-cache-remove pytest-cache-remove

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: build-remove cache-remove
