APP ?= playground
PORT ?= 8000
HOST ?= 127.0.0.1

.PHONY: help setup new-app list db-up db-down dev migrate revision check seed test lint

help:
	@python3 scripts/manage.py --help

setup:
	@python3 scripts/manage.py init-env
	uv sync --all-packages

new-app:
	python3 scripts/manage.py new "$(NAME)"
	uv sync --all-packages

list:
	@python3 scripts/manage.py list

db-up:
	docker compose up -d --wait postgres

db-down:
	docker compose down

dev:
	python3 scripts/manage.py dev "$(APP)" --port "$(PORT)" --host "$(HOST)"

migrate:
	python3 scripts/manage.py migrate "$(APP)"

revision:
	python3 scripts/manage.py revision "$(APP)" --message "$(MESSAGE)"

check:
	python3 scripts/manage.py check "$(APP)"

# This seed is specific to the playground's Trailhead store.
seed:
	uv run --package ai-app-playground python -m ai_playground.seed

test:
	uv run --all-packages pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
