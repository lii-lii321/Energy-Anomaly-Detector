import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import multivariate_normal

import main


@pytest.fixture()
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_root_reports_service_is_running(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Energy Monitoring System is Running"}


def test_predict_returns_non_anomaly_for_a_typical_point(client):
    sample = {"pressure": float(main.mu[0]), "current": float(main.mu[1])}

    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    prediction = body["prediction"]
    assert prediction["is_anomaly"] is False
    assert prediction["threshold"] == main.epsilon
    assert prediction["probability"] >= main.epsilon
    assert body["message"] == "系统运行正常"


def test_predict_probability_matches_scipy_reference(client):
    sample = {"pressure": 2.05, "current": 14.8}

    response = client.post("/predict", json=sample)

    expected = multivariate_normal.pdf(
        [2.05, 14.8], mean=main.mu, cov=main.sigma
    )
    assert response.json()["prediction"]["probability"] == pytest.approx(
        expected, rel=1e-12
    )


def test_predict_flags_logical_conflict_as_anomaly(client):
    sample = {"pressure": 1.70, "current": 18.5}

    response = client.post("/predict", json=sample)

    prediction = response.json()["prediction"]
    expected = multivariate_normal.pdf([1.70, 18.5], mean=main.mu, cov=main.sigma)
    assert prediction["probability"] == pytest.approx(expected, rel=1e-12)
    assert prediction["is_anomaly"] is (prediction["probability"] < main.epsilon)
    if prediction["is_anomaly"]:
        assert response.json()["message"] == "检测到异常运行"


def test_predict_rejects_missing_and_non_numeric_fields(client):
    assert client.post("/predict", json={"pressure": 2.1}).status_code == 422
    assert (
        client.post(
            "/predict", json={"pressure": "high", "current": 15.0}
        ).status_code
        == 422
    )


def test_predict_returns_500_when_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(main, "mu", None)

    response = client.post("/predict", json={"pressure": 2.1, "current": 15.0})

    assert response.status_code == 500
    assert "模型未加载" in response.json()["detail"]


def test_loaded_model_parameters_are_two_dimensional():
    assert isinstance(main.mu, np.ndarray)
    assert main.sigma.shape == (2, 2)
