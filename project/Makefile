# ============================================================================
# Insole Foot Classification — Makefile
# ----------------------------------------------------------------------------
# Common entrypoints: install, run, train, test, lint, docker, build.
#   $ make help
# ============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

# Override at the CLI:   make train EPOCHS=20
EPOCHS     ?= 50
BATCH_SIZE ?= 16
DATA_DIR   ?= data

.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / \
		{ printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ----------------------------------------------------------------------------
# Local environment
# ----------------------------------------------------------------------------
.PHONY: venv
venv: ## Create a virtualenv at .venv
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: venv ## Install runtime dependencies into .venv
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: venv ## Install dev + runtime dependencies
	$(PIP) install -r requirements-dev.txt

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
.PHONY: app
app: ## Launch the desktop GUI
	$(PY) -m app.main

.PHONY: api
api: ## Run the FastAPI server with hot-reload
	$(PY) -m uvicorn backend.server.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: train
train: ## Train the model (override EPOCHS, BATCH_SIZE, DATA_DIR)
	$(PY) scripts/train.py --data-dir $(DATA_DIR) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

.PHONY: predict
predict: ## Predict on three images (set LATERAL=… TOP=… BACK=…)
	$(PY) scripts/predict.py --lateral $(LATERAL) --top $(TOP) --back $(BACK)

.PHONY: prepare-data
prepare-data: ## Scan the data folder and print summary
	$(PY) scripts/prepare_data.py --data-dir $(DATA_DIR) --output-csv data/processed/manifest.csv

.PHONY: seed-db
seed-db: ## Insert demo patients + a sample classification
	$(PY) scripts/seed_demo_data.py

# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
.PHONY: migrate
migrate: ## Apply database migrations
	$(VENV)/bin/alembic upgrade head

.PHONY: migration
migration: ## Create a new auto-generated migration (NAME=description)
	$(VENV)/bin/alembic revision --autogenerate -m "$(NAME)"

# ----------------------------------------------------------------------------
# Quality
# ----------------------------------------------------------------------------
.PHONY: test
test: ## Run the test suite
	$(VENV)/bin/pytest tests/ -v

.PHONY: lint
lint: ## Lint with ruff + format-check with black
	$(VENV)/bin/ruff check app backend scripts
	$(VENV)/bin/black --check app backend scripts

.PHONY: format
format: ## Auto-format with black + ruff --fix
	$(VENV)/bin/black app backend scripts
	$(VENV)/bin/ruff check --fix app backend scripts

# ----------------------------------------------------------------------------
# Build / Package
# ----------------------------------------------------------------------------
.PHONY: build-exe
build-exe: ## Build a standalone desktop binary via PyInstaller
	$(PY) app/build_exe.py

# ----------------------------------------------------------------------------
# Docker
# ----------------------------------------------------------------------------
.PHONY: docker-up
docker-up: ## Start db + backend in containers
	docker compose up -d --build db backend

.PHONY: docker-down
docker-down: ## Stop all containers
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail backend logs
	docker compose logs -f backend

.PHONY: docker-train
docker-train: ## Run a training job inside docker
	docker compose --profile training run --rm trainer \
		--data-dir /workspace/data --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

.PHONY: docker-shell
docker-shell: ## Open a shell inside the backend container
	docker compose exec backend bash

.PHONY: clean
clean: ## Remove build artefacts and caches
	rm -rf build dist *.spec .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
