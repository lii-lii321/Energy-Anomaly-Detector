import pytest
from fastapi.testclient import TestClient

import main
from sensor_limits import (
    CURRENT_LIMIT_MAX,
    CURRENT_LIMIT_MIN,
    PRESSURE_LIMIT_MAX,
    PRESSURE_LIMIT_MIN,
)


@pytest.fixture()
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def _sample(pressure, current):
    return {"pressure": pressure, "current": current}


def test_model_info_reports_metadata_and_shared_limits(client):
    response = client.get("/model")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["epsilon"] == main.epsilon
    assert body["n_samples"] == 201
    assert body["threshold_method"] == "f1"
    assert body["trained_at"]
    assert body["feature_limits"]["pressure"] == {
        "min": PRESSURE_LIMIT_MIN,
        "max": PRESSURE_LIMIT_MAX,
        "unit": "MPa",
    }
    assert body["feature_limits"]["current"] == {
        "min": CURRENT_LIMIT_MIN,
        "max": CURRENT_LIMIT_MAX,
        "unit": "A",
    }


def test_model_info_does_not_leak_model_path(client):
    body = client.get("/model").json()

    text = str(body)
    assert "models" not in text
    assert ".pkl" not in text


def test_model_info_reports_unloaded_model(client, monkeypatch):
    monkeypatch.setattr(main, "mu", None)

    body = client.get("/model").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["epsilon"] is None
    assert body["n_samples"] is None


def test_batch_predict_scores_multiple_points_with_summary(client):
    points = [_sample(2.05, 14.8), _sample(2.15, 15.2), _sample(1.70, 18.5)]

    response = client.post("/predict/batch", json={"samples": points})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["n_total"] == 3
    assert body["n_anomaly"] == sum(p["is_anomaly"] for p in body["predictions"])
    assert [p["pressure"] for p in body["predictions"]] == [2.05, 2.15, 1.70]
    assert all(p["threshold"] == main.epsilon for p in body["predictions"])


def test_batch_predictions_match_single_endpoint(client):
    points = [_sample(2.05, 14.8), _sample(1.70, 18.5)]

    batch_body = client.post("/predict/batch", json={"samples": points}).json()
    singles = [
        client.post("/predict", json=point).json()["prediction"] for point in points
    ]

    for item, single in zip(batch_body["predictions"], singles, strict=True):
        assert item["probability"] == pytest.approx(single["probability"], rel=1e-12)
        assert item["is_anomaly"] == single["is_anomaly"]


def test_batch_predict_rejects_empty_and_oversized_batches(client):
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422

    oversized = {"samples": [_sample(2.1, 15.0)] * (main.BATCH_MAX_SAMPLES + 1)}
    assert client.post("/predict/batch", json=oversized).status_code == 422


def test_batch_predict_rejects_sample_outside_physical_limits(client):
    response = client.post(
        "/predict/batch",
        json={"samples": [_sample(2.1, 15.0), _sample(-1.0, 15.0)]},
    )

    assert response.status_code == 422


def test_batch_predict_returns_500_when_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(main, "mu", None)

    response = client.post(
        "/predict/batch", json={"samples": [_sample(2.1, 15.0)]}
    )

    assert response.status_code == 500
    assert "模型未加载" in response.json()["detail"]
