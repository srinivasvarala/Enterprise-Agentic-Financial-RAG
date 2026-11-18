.PHONY: install build up down logs migrate dev lint fmt test test-int eval

install:
	uv sync
	uv run pre-commit install

build:
	docker compose build

up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f --tail=100 api worker

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
