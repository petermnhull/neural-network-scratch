#!make
include .env
export

pypi-init:
	poetry source add testpypi https://test.pypi.org/legacy/
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry config pypi-token.testpypi ${PYPI_TOKEN}

pypi-publish:
	poetry publish -r testpypi

poetryconf:
	poetry config virtualenvs.create false

poetryinstall:
	poetry install --no-root

fmt:
	ruff check . --fix
	ruff format . 

test:
	coverage run --source neural_network_scratch/ -m pytest tests/
	coverage report
