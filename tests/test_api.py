from fastapi.testclient import TestClient
import numpy as np
import pytest
from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


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
