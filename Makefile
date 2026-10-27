# BookMind Pro — developer shortcuts (Linux / macOS / WSL / Git Bash).
# Windows PowerShell users: see scripts/setup.ps1 and the README.

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev lint format typecheck test cov check download \
        pipeline pipeline-offline train-knn evaluate run serve docker-build docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: venv  ## Install runtime deps + package
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

install-dev: venv  ## Install everything for development
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps
	$(BIN)/pre-commit install || true
	@test -f .env || cp .env.example .env

lint:  ## Ruff lint
	$(BIN)/ruff check src tests

format:  ## Auto-format code
	$(BIN)/ruff format src tests
	$(BIN)/ruff check src tests --fix

typecheck:  ## mypy (strict)
	$(BIN)/mypy

test:  ## Run the test suite
	$(BIN)/pytest

cov:  ## Tests with HTML coverage report (htmlcov/index.html)
	$(BIN)/pytest --cov --cov-report=term --cov-report=html

check: lint typecheck cov  ## Everything CI runs

download:  ## Download both Kaggle datasets (kagglehub)
	$(BIN)/bookmind-pipeline download

pipeline:  ## Full pipeline: Ollama genres + MiniLM index + k-NN
	$(BIN)/bookmind-pipeline all

train-knn:  ## Train only the collaborative-filtering model
	$(BIN)/bookmind-pipeline train-knn

evaluate:  ## Evaluate k-NN (Hit-Rate@5, coverage) for all rating modes
	$(BIN)/bookmind-pipeline evaluate-knn --compare-modes

pipeline-offline:  ## No Ollama, no model download (keyword genres + hashing encoder + k-NN)
	$(BIN)/bookmind-pipeline all --classifier keyword --encoder hashing

run:  ## Dev server with auto-reload on :8000
	BOOKMIND_DEBUG=true $(PY) -m bookmind

serve:  ## Production server (gunicorn + uvicorn workers)
	$(BIN)/gunicorn -c gunicorn.conf.py "bookmind.main:create_app()"

docker-build:  ## Build the Docker image
	docker build --build-arg PRELOAD_MODEL=true -t bookmind:latest .

docker-up:  ## Start web + redis with docker compose
	docker compose up --build -d

docker-down:  ## Stop docker compose services
	docker compose down

clean:  ## Remove caches and build outputs
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
