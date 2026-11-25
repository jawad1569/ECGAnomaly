"""Model loader for the ECG API.

Where the model comes from is set by MODEL_SOURCE:
  - "auto" (default): MLflow registry first, then the newest models/ecg_model_*.joblib
  - "registry": MLflow registry only
  - "file": newest models/ecg_model_*.joblib only (used in Docker and CI, where there is
    no mlflow.db)

MLflow is imported only when the registry is used, so the API image does not need it.
The model is cached in memory after the first load.
"""

import glob
import os
import warnings
from typing import List, Optional

import joblib
import numpy as np

warnings.filterwarnings("ignore")

_model: Optional[object] = None
_class_names: Optional[List[str]] = None
_model_version: str = "unknown"

N_LEADS = 12
DEFAULT_CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
MODEL_DIR = os.getenv("MODEL_DIR", "models")


def extract_features_from_signal(signal_array: np.ndarray) -> np.ndarray:
    """Extract rich v2 features (must match training)."""
    from src.utils.features import extract_features

    return extract_features(signal_array)


def _setup_tracking(tracking_uri: Optional[str] = None):
    import mlflow

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    return uri


def model_exists_in_registry(
    model_name: str = "ecg_model_v1",
    tracking_uri: Optional[str] = None,
) -> bool:
    """Check if model exists in the MLflow model registry."""
    try:
        import mlflow

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
    import mlflow
    import mlflow.sklearn

    global _model, _class_names, _model_version
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
        _class_names = list(DEFAULT_CLASSES)

    _model_version = f"{model_name}/v{latest_version.version}"
    return _model, _class_names


def latest_model_file(model_dir: Optional[str] = None) -> Optional[str]:
    """Return the newest models/ecg_model_<timestamp>_<est>.joblib, or None.

    File names start with a sortable timestamp, so the last one by name is the newest.
    """
    files = sorted(glob.glob(os.path.join(model_dir or MODEL_DIR, "ecg_model_*.joblib")))
    return files[-1] if files else None


def load_model_from_file(path: Optional[str] = None) -> tuple:
    """Load a model saved by train.py ({"model": ..., "classes": [...]})."""
    global _model, _class_names, _model_version
    path = path or latest_model_file()
    if path is None:
        raise RuntimeError(f"No ecg_model_*.joblib file found in '{MODEL_DIR}/'.")

    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        _model = obj["model"]
        _class_names = [str(c) for c in (obj.get("classes") or DEFAULT_CLASSES)]
    else:  # a bare estimator
        _model = obj
        _class_names = list(DEFAULT_CLASSES)

    _model_version = os.path.splitext(os.path.basename(path))[0]
    return _model, _class_names


def get_model(raise_on_error: bool = True):
    """Return the cached model, loading it on first use (see MODEL_SOURCE above)."""
    global _model, _class_names

    if _model is None:
        source = os.getenv("MODEL_SOURCE", "auto").lower()
        model_name = os.getenv("MLFLOW_MODEL_NAME", "ecg_model_v1")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        errors = []

        if source in ("auto", "registry"):
            try:
                if model_exists_in_registry(model_name, tracking_uri):
                    load_model_from_registry(model_name, tracking_uri)
                else:
                    errors.append(f"model '{model_name}' not in MLflow registry")
            except Exception as e:
                errors.append(f"registry: {e}")

        if _model is None and source in ("auto", "file"):
            try:
                load_model_from_file()
            except Exception as e:
                errors.append(f"file: {e}")

        if _model is None:
            if raise_on_error:
                raise RuntimeError("Could not load model: " + "; ".join(errors))
            return None, None

    return _model, _class_names


def get_model_version() -> str:
    """Identifier of the loaded model (registry version or file name)."""
    return _model_version


def reset_model_cache():
    """Forget the cached model so the next get_model() reloads it."""
    global _model, _class_names, _model_version
    _model, _class_names, _model_version = None, None, "unknown"


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
        "model_version": get_model_version(),
    }
