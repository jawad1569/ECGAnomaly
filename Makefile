.PHONY: setup data train dev test lint format clean check docker-build docker-run docker-stop

IMAGE ?= ecg-api
TAG ?= latest

setup:
	pip install -r requirements-dev.txt

# Step 1: validate PTB-XL (dataset/) and build features
data:
	python src/pipeline/prepare_data.py

# Step 2: AutoML training, evaluation, model saving, MLflow logging
train:
	python src/pipeline/train.py

# API on http://127.0.0.1:8000/docs
dev:
	python -m uvicorn src.api.main:app --reload

test:
	pytest -v

# Docker: build the API image and run it on http://localhost:8000
docker-build:
	docker build -t $(IMAGE):$(TAG) .

docker-run:
	docker run -d --name ecg-api -p 8000:8000 $(IMAGE):$(TAG)

docker-stop:
	docker rm -f ecg-api

lint:
	ruff check .

format:
	black .

check:
	pre-commit run --all-files

clean:
	python -c "import shutil, glob; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True) + glob.glob('**/.pytest_cache', recursive=True)]"
