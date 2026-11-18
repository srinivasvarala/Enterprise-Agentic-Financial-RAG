.PHONY: install up down migrate dev lint fmt test test-int eval

install:
	uv sync
	uv run pre-commit install

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	uv run alembic upgrade head

dev:
	uv run uvicorn finrag.api.app:app --reload --port 8000

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-int:
	uv run pytest -m integration

eval:
	uv run pytest -m eval evals/
