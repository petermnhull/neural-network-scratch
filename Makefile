#!make
include .env
export

poetryconf:
	poetry config virtualenvs.create false

poetryinstall:
	poetry install --no-root

fmt:
	ruff check . --fix
	ruff format . 
