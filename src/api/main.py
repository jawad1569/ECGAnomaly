"""FastAPI application entry point for the ECG Anomaly Detection API (Milestone 1)."""

# ruff: noqa: E402
import logging
import os
import warnings

from dotenv import load_dotenv

from src.utils.model_loader import get_model

load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

from fastapi import FastAPI

from src.api import monitoring
from src.api.routers import health, predict

app = FastAPI(
    title="ECG Anomaly Detection API",
    description="Classify ECG signals as Normal or Abnormal using an MLflow-registered model.",
    version="1.0.0",
)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

app.middleware("http")(monitoring.track_requests)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(monitoring.router)


@app.on_event("startup")
async def startup_event():
    """Initialize and load the model on startup."""
    print("=" * 60)
    print("ECG Anomaly Detection API Starting...")
    print("=" * 60)

    # Training on startup needs the full dataset and several minutes, so it is opt-in
    # (AUTO_TRAIN=true). Docker and CI load the committed models/*.joblib instead.
    auto_train = os.getenv("AUTO_TRAIN", "false").lower() in ("1", "true", "yes")

    from src.utils.model_loader import get_model_version, reset_model_cache

    try:
        model_exists = get_model(raise_on_error=False)[0] is not None

        if not model_exists and auto_train:
            print("=" * 60)
            print("Model not found in registry.")
            print("Starting automatic training...")
            print("This may take several minutes. Please wait...")
            print("=" * 60)

            training_success = False
            try:
                from src.pipeline.train import main as train_main

                train_main()
                training_success = True

            except (MemoryError, OSError) as mem_error:
                print(f"Training failed due to memory/OS error: {mem_error}")
                print("Try: python manage.py train")
            except FileNotFoundError as data_error:
                print(f"Training data not found: {data_error}")
                print("Ensure training data exists in data/raw/")
            except Exception as train_error:
                print(f"Training failed: {train_error}")
                print("Try: python manage.py train")

            if training_success:
                print("Training completed successfully!")
                reset_model_cache()
    except Exception as check_error:
        print(f"Could not check for a model: {check_error}")

    # Load model into memory
    try:
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            model, class_names = get_model(raise_on_error=False)
        finally:
            sys.stdout = old_stdout

        if model is not None:
            print(f"Model ready ({get_model_version()}). Classes: {class_names}")
        else:
            print("No model found. Run: python manage.py train (or set AUTO_TRAIN=true)")
    except Exception as e:
        print(f"Model will load on first request: {e}")

    print("=" * 60)
    print("API ready. Docs at /docs, health at /health")
    print("=" * 60)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "ECG Anomaly Detection API", "tracking_uri": mlflow_uri}
