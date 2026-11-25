import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.monitoring import RequestStats, stats

client = TestClient(app)


def test_stats_counts_requests_and_latency():
    stats.reset()
    for _ in range(3):
        assert client.get("/health").status_code == 200
    signal = np.random.randn(500, 12).tolist()
    assert client.post("/predict", json={"signal": signal}).status_code == 200

    data = client.get("/stats").json()
    assert data["request_count"] == 4
    assert data["avg_latency_ms"] > 0
    assert data["per_endpoint"]["GET /health"]["count"] == 3
    assert data["per_endpoint"]["POST /predict"]["count"] == 1
    assert data["model_version"].startswith("ecg_model_")


def test_stats_itself_is_not_counted():
    stats.reset()
    client.get("/stats")
    client.get("/metrics")
    assert client.get("/stats").json()["request_count"] == 0


def test_ui_assets_are_not_counted():
    stats.reset()
    client.get("/static/app.js")
    client.get("/static/styles.css")
    assert client.get("/stats").json()["request_count"] == 0


def test_stats_counts_client_errors():
    stats.reset()
    bad = np.random.randn(100, 6).tolist()  # 6 leads instead of 12
    assert client.post("/predict", json={"signal": bad}).status_code == 400
    data = client.get("/stats").json()
    assert data["client_errors_4xx"] == 1
    assert data["server_errors_5xx"] == 0


def test_request_stats_average():
    s = RequestStats()
    s.record("GET", "/health", 200, 10.0)
    s.record("GET", "/health", 200, 30.0)
    snap = s.snapshot()
    assert snap["request_count"] == 2
    assert snap["avg_latency_ms"] == 20.0
    assert snap["per_endpoint"]["GET /health"]["max_latency_ms"] == 30.0
