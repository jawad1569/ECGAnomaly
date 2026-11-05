.PHONY: setup data dev test lint format docker clean run check audit train evidently ingest rag ui ui-build

# Development server
dev:
	python -m uvicorn src.api.main:app --reload

# Run tests
test:
	pytest -v

# Run linters
lint:
	ruff check .
	black --check .

# Format code
format:
	black .

# Docker commands
docker:
	docker build -t ecg-app .

run:
	docker run -p 8000:8000 ecg-app

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Pre-commit checks
check:
	pre-commit run --all-files

# Security audit
audit:
	bash scripts/dependency_audit.sh

# Training pipeline
train:
	python src/pipeline/train.py

# Evidently dashboard server (port 7000)
evidently:
	python src/monitoring/evidently_server.py

# Ingest PDFs and create vector database for RAG
ingest:
	python src/ingest.py

# Alias for ingest
rag:
	python src/ingest.py

# Frontend commands
ui:
	cd ui && npm run dev

ui-build:
	cd ui && npm run build

# Milestone 1 helpers
setup:
	pip install -r requirements.txt

# Reformat a PTB-XL sample into data/raw CSVs (needs dataset/ from PhysioNet)
data:
	python src/pipeline/prepare_data.py
