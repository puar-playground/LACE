.PHONY: setup
setup:
	pip install -U pip setuptools wheel poetry
	poetry install

.PHONY: format
format:
	poetry run ruff format --check --diff .

.PHONY: lint
lint:
	poetry run ruff check --output-format=github .

.PHONY: typecheck
typecheck:
	poetry run mypy .

.PHONY: test
test:
	poetry run pytest
