# ============================================================
# Makefile — DeepRegime Development Commands
# ============================================================

.PHONY: install setup train serve dashboard api test clean docker-up docker-down

# ── Setup ──────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

setup: install
	cp -n .env.example .env || true
	python database/migrations.py init
	@echo "✅ Setup complete. Edit .env to add API keys."

# ── Training ───────────────────────────────────────────────
train:
	python train_model.py --start 2015-01-01

train-quick:
	python train_model.py --start 2020-01-01 --hmm-only

train-full:
	python train_model.py --start 2015-01-01 --no-eval

# ── Running ────────────────────────────────────────────────
serve:
	python app.py serve

api:
	python app.py serve --api-only

dashboard:
	python app.py serve --dash-only

api-dev:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Testing ────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=. --cov-report=html

test-fast:
	pytest tests/ -v -x -k "not slow"

# ── Docker ─────────────────────────────────────────────────
docker-up:
	cd docker && docker-compose up --build -d
	@echo "✅ Stack running"
	@echo "   Dashboard: http://localhost:8501"
	@echo "   API:       http://localhost:8000/api/docs"

docker-down:
	cd docker && docker-compose down

docker-logs:
	cd docker && docker-compose logs -f

# ── Database ───────────────────────────────────────────────
db-init:
	python database/migrations.py init

db-reset:
	python database/migrations.py reset

# ── Notebook ───────────────────────────────────────────────
notebook:
	jupyter lab notebooks/

# ── Cleanup ────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "✅ Cleaned"

clean-models:
	rm -rf models/saved/*.pkl models/saved/*.keras models/saved/*.json
	@echo "✅ Saved models deleted"

# ── Health Check ───────────────────────────────────────────
check:
	python app.py check

# ── Help ───────────────────────────────────────────────────
help:
	@echo ""
	@echo "  DeepRegime — Available Commands"
	@echo "  ================================"
	@echo "  make setup        — Install deps + init DB"
	@echo "  make train        — Train HMM + LSTM (2015→now)"
	@echo "  make train-quick  — Quick HMM-only train (2020→now)"
	@echo "  make serve        — Run API + Dashboard"
	@echo "  make api          — Run API only"
	@echo "  make dashboard    — Run dashboard only"
	@echo "  make notebook     — Launch Jupyter Lab"
	@echo "  make docker-up    — Launch full Docker stack"
	@echo "  make docker-down  — Stop Docker stack"
	@echo "  make test         — Run test suite"
	@echo "  make clean        — Remove cache files"
	@echo ""
