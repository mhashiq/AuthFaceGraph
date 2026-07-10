# ================================================================
# AuthBrain AI Face Analysis Engine — Developer Makefile
# ================================================================

.PHONY: help setup dev backend frontend test lint clean models

PYTHON   = python3
PIP      = pip3
VENV     = .venv
UVICORN  = $(VENV)/bin/uvicorn
PYTEST   = $(VENV)/bin/pytest

# ── Default target ──────────────────────────────────────────────
help:
	@echo ""
	@echo "  AuthBrain AI Face Analysis Engine"
	@echo "  ─────────────────────────────────"
	@echo ""
	@echo "  make setup       Install all dependencies (backend + frontend)"
	@echo "  make dev         Start backend + frontend in dev mode"
	@echo "  make backend     Start FastAPI backend only"
	@echo "  make frontend    Start React frontend only"
	@echo "  make test        Run all backend tests"
	@echo "  make test-unit   Run unit tests only"
	@echo "  make lint        Run Python linters"
	@echo "  make models      Download MediaPipe models"
	@echo "  make migrate     Run database migrations"
	@echo "  make docker-up   Start all services with Docker Compose"
	@echo "  make docker-down Stop all Docker services"
	@echo "  make clean       Clean build artifacts"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────

setup: setup-backend setup-frontend
	@echo "✅ Setup complete"

setup-backend:
	@echo "→ Installing backend dependencies..."
	cd backend && $(PYTHON) -m venv .venv && \
	  .venv/bin/pip install --upgrade pip && \
	  .venv/bin/pip install -r requirements.txt
	@echo "✅ Backend dependencies installed"

setup-frontend:
	@echo "→ Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✅ Frontend dependencies installed"

# ── Environment ─────────────────────────────────────────────────

env:
	@if [ ! -f .env ]; then \
	  cp .env.example .env; \
	  echo "✅ .env created from .env.example — please update the values"; \
	else \
	  echo "ℹ️  .env already exists"; \
	fi

# ── Models ──────────────────────────────────────────────────────

models:
	@echo "→ Downloading MediaPipe Face Landmarker model..."
	mkdir -p models
	curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
	  -o models/face_landmarker.task
	@echo "✅ Model downloaded to models/face_landmarker.task"

# ── Development ─────────────────────────────────────────────────

dev:
	@echo "→ Starting backend + frontend..."
	@$(MAKE) -j2 backend frontend

backend:
	@echo "→ Starting FastAPI backend on :8000..."
	cd backend && .venv/bin/uvicorn app.main:app \
	  --host 0.0.0.0 --port 8000 \
	  --reload --log-level debug

frontend:
	@echo "→ Starting Vite frontend on :3000..."
	cd frontend && npm run dev

# ── Database ─────────────────────────────────────────────────────

migrate:
	@echo "→ Running database migrations..."
	cd backend && .venv/bin/python -m alembic upgrade head

migrate-create:
	cd backend && .venv/bin/alembic revision --autogenerate -m "$(msg)"

# ── Testing ──────────────────────────────────────────────────────

test:
	@echo "→ Running all tests..."
	cd backend && $(PYTEST) tests/ -v --tb=short

test-unit:
	cd backend && $(PYTEST) tests/unit/ -v --tb=short

test-integration:
	cd backend && $(PYTEST) tests/integration/ -v --tb=short

test-coverage:
	cd backend && $(PYTEST) tests/ \
	  --cov=app \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  -v

# ── Code Quality ─────────────────────────────────────────────────

lint:
	@echo "→ Running linters..."
	cd backend && .venv/bin/ruff check app/ tests/
	cd backend && .venv/bin/mypy app/ --ignore-missing-imports

format:
	cd backend && .venv/bin/ruff format app/ tests/
	cd backend && .venv/bin/ruff check app/ tests/ --fix

# ── Docker ───────────────────────────────────────────────────────

docker-up:
	docker compose up --build -d
	@echo "✅ Services started. Backend: http://localhost:8000/api/docs"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f backend

docker-prod:
	docker compose --profile production up --build -d

# ── Cleanup ──────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf backend/.venv frontend/node_modules frontend/dist
	rm -rf backend/htmlcov backend/.coverage
	@echo "✅ Cleanup complete"
