"""FastAPI application entry point for the ECG Anomaly Detection API (Milestone 1)."""

# ruff: noqa: E402
from dotenv import load_dotenv
import os
import warnings
import logging
from src.utils.model_loader import get_model

load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)

from fastapi import FastAPI
from src.api.routers import health, predict

app = FastAPI(
    title="ECG Anomaly Detection API",
    description="Classify ECG signals as Normal or Abnormal using an MLflow-registered model.",
    version="1.0.0",
)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

app.include_router(health.router)
app.include_router(predict.router)


@app.on_event("startup")
async def startup_event():
    """Initialize and load the model on startup."""
    print("=" * 60)
    print("ECG Anomaly Detection API Starting...")
    print("=" * 60)

    model_name   = os.getenv("MLFLOW_MODEL_NAME", "ecg_model_v1")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    from src.utils.model_loader import model_exists_in_registry

    try:
        model_exists = model_exists_in_registry(
            model_name=model_name, tracking_uri=tracking_uri
        )

        if not model_exists:
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
                import src.utils.model_loader as model_loader_module
                model_loader_module._model = None
                model_loader_module._class_names = None
    except Exception as check_error:
        print(f"Could not check model registry: {check_error}")

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
            print(f"Model ready. Classes: {class_names}")
        else:
            print("Model not loaded yet. Run: python manage.py train")
    except Exception as e:
        print(f"Model will load on first request: {e}")

    print("=" * 60)
    print("API Ready! Listening on http://127.0.0.1:8000")
    print("=" * 60)


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "ECG Anomaly Detection API", "tracking_uri": mlflow_uri}
