#* Variables
SHELL ?= /usr/bin/env bash
PYTHON ?= python
PYTHONPATH := `pwd`
POETRY ?= poetry

#* Poetry
.PHONY: poetry-install
poetry-install:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

.PHONY: poetry-plugins
poetry-plugins:
	$(POETRY) self add poetry-plugin-up

.PHONY: poetry-env
poetry-env:
	$(POETRY) config virtualenvs.in-project true

.PHONY: publish
publish:
	$(POETRY) publish --build

#* Installation
.PHONY: install
install: poetry-env
	$(POETRY) lock -n && poetry export --without-hashes > requirements-lock.txt
	$(POETRY) install -n
	-poetry run mypy --install-types --non-interactive ./

.PHONY: install-dev
install-dev: poetry-env install
	$(POETRY) install -n --with dev

.PHONY: pre-commit-install
pre-commit-install:
	$(POETRY) run pre-commit install


#* Formatters
.PHONY: codestyle
codestyle:
	$(POETRY) run ruff format --config=pyproject.toml stac_model tests

.PHONY: format
format: codestyle

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=stac_model tests/

.PHONY: check
check: check-examples check-markdown check-lint check-mypy check-safety check-citation

.PHONY: check-all
check-all: check

.PHONY: mypy
mypy:
	$(POETRY) run mypy --config-file pyproject.toml ./

.PHONY: check-mypy
check-mypy: mypy

# NOTE:
#  purposely running with docker rather than python package due to conflicting dependencies
#  see https://github.com/citation-file-format/cffconvert/issues/292
.PHONY: check-citation
check-citation:
	docker run --rm -v $(PYTHONPATH)/CITATION.cff:/app/CITATION.cff citationcff/cffconvert --validate

.PHONY: check-safety
check-safety:
	$(POETRY) check
	$(POETRY) run safety check --full-report
	$(POETRY) run bandit -ll --recursive stac_model tests

.PHONY: lint
lint:
	$(POETRY) run ruff --config=pyproject.toml ./
	$(POETRY) run pydocstyle --count --config=pyproject.toml ./
	$(POETRY) run pydoclint --config=pyproject.toml ./

.PHONY: check-lint
check-lint: lint

.PHONY: format-lint
format-lint:
	$(POETRY) run ruff --config=pyproject.toml --fix ./

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
	$(POETRY) up --only=dev-dependencies --latest

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
