.PHONY: setup data train dev test lint format clean check

setup:
	pip install -r requirements.txt

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

lint:
	ruff check .
	black --check .

format:
	black .

check:
	pre-commit run --all-files

clean:
	python -c "import shutil, glob; [shutil.rmtree(d, ignore_errors=True) for d in glob.glob('**/__pycache__', recursive=True) + glob.glob('**/.pytest_cache', recursive=True)]"
