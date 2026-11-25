import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    assert "message" in response.json()


def test_web_ui_is_served():
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "ECG Anomaly Detection" in page.text
    for asset in ["/static/app.js", "/static/styles.css"]:
        assert client.get(asset).status_code == 200


def test_model_info_reports_accuracy_and_date():
    info = client.get("/model").json()
    assert info["model_version"].startswith("ecg_model_")
    assert 0.0 < info["accuracy"] <= 1.0
    assert 0.0 < info["f1_macro"] <= 1.0
    assert info["trained_at"].startswith("20")  # ISO date, e.g. 2026-11-06T01:40:53


def test_generated_test_samples_are_single_labelled_ecgs():
    import glob

    import pandas as pd

    files = sorted(glob.glob("samples/test_ecgs/*_*.csv"))
    assert len(files) >= 5
    for f in files:
        df = pd.read_csv(f)
        assert df["ecg_id"].nunique() == 1, f
        assert df["label"].iloc[0] in ["CD", "HYP", "MI", "NORM", "STTC"], f
        assert df.shape == (1000, 14), f  # ecg_id, label, 12 channels


def test_ui_sample_is_a_valid_request():
    sample = client.get("/static/sample_ecg_1.json").json()
    assert len(sample["signal"]) == 1000 and len(sample["signal"][0]) == 12
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["CD", "HYP", "MI", "NORM", "STTC"]


def test_predict_endpoint_valid():
    signal = np.random.randn(100, 12).tolist()
    payload = {"signal": signal, "ecg_id": 101}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "explanation" in data


def test_predict_endpoint_invalid_shape():
    # Only 6 channels instead of 12
    signal = np.random.randn(100, 6).tolist()
    payload = {"signal": signal}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_explain_endpoint():
    signal = np.random.randn(100, 12).tolist()
    payload = {"signal": signal, "ecg_id": 102}
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "lead_importance" in data
    assert len(data["lead_importance"]) == 12
