"""Model loader utility for MLflow-registered models.

Loads the registered model (scikit-learn compatible) from the MLflow Model Registry
and caches it in memory for fast API inference.
"""

import os
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
from typing import Optional, List

warnings.filterwarnings("ignore")

_model: Optional[object] = None
_class_names: Optional[List[str]] = None

N_LEADS = 12


def extract_features_from_signal(signal_array: np.ndarray) -> np.ndarray:
    """Extract rich v2 features (must match training)."""
    from src.utils.features import extract_features

    return extract_features(signal_array)


def _setup_tracking(tracking_uri: Optional[str] = None):
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    return uri


def model_exists_in_registry(
    model_name: str = "ecg_model_v1",
    tracking_uri: Optional[str] = None,
) -> bool:
    """Check if model exists in the MLflow model registry."""
    try:
        _setup_tracking(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            versions = client.search_model_versions(f"name='{model_name}'")
        return len(versions) > 0
    except Exception:
        return False


def load_model_from_registry(
    model_name: str = "ecg_model_v1",
    tracking_uri: Optional[str] = None,
) -> tuple:
    """Load latest version of registered model and class names."""
    global _model, _class_names
    _setup_tracking(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise RuntimeError(f"No registered versions found for model '{model_name}'.")

    latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    model_uri = f"models:/{model_name}/{latest_version.version}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model = mlflow.sklearn.load_model(model_uri)

    # Load class names from run artifact if present
    try:
        cn_path = mlflow.artifacts.download_artifacts(
            run_id=latest_version.run_id, artifact_path="class_names.txt"
        )
        with open(cn_path, "r") as f:
            _class_names = [line.strip() for line in f if line.strip()]
    except Exception:
        _class_names = ["CD", "HYP", "MI", "NORM", "STTC"]

    return _model, _class_names


def get_model(raise_on_error: bool = True):
    """Retrieve cached model or load from registry."""
    global _model, _class_names

    if _model is None:
        model_name = os.getenv("MLFLOW_MODEL_NAME", "ecg_model_v1")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        try:
            _model, _class_names = load_model_from_registry(
                model_name=model_name, tracking_uri=tracking_uri
            )
        except Exception as e:
            if raise_on_error:
                raise RuntimeError(f"Could not load model '{model_name}': {e}") from e
            return None, None

    return _model, _class_names


def predict_ecg_signal(signal_data: np.ndarray) -> dict:
    """Predict ECG class from 12-channel signal."""
    model, class_names = get_model()

    if signal_data.ndim == 3:
        signal_data = signal_data[0]

    features = extract_features_from_signal(signal_data).reshape(1, -1)

    # Some models support predict_proba, others decision_function
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        probabilities = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
    else:
        pred = model.predict(features)[0]
        idx = int(pred)
        confidence = 1.0
        probabilities = {name: (1.0 if i == idx else 0.0) for i, name in enumerate(class_names)}

    return {
        "predicted_class": class_names[idx],
        "confidence": confidence,
        "probabilities": probabilities,
        "class_index": idx,
        "model_version": "v1",
    }
