import os

# Tests use the committed models/ecg_model_*.joblib, so they run the same on a fresh
# clone and in CI (no mlflow.db there). Override with MODEL_SOURCE=auto to test the registry.
os.environ.setdefault("MODEL_SOURCE", "file")
os.environ.setdefault("AUTO_TRAIN", "false")
