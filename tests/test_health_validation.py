from pathlib import Path

import annotated_types
import pytest
from fastapi.testclient import TestClient

import main
from conftest import REPO_ROOT
from sensor_limits import (
    CURRENT_LIMIT_MAX,
    CURRENT_LIMIT_MIN,
    CURRENT_SLIDER_MAX,
    CURRENT_SLIDER_MIN,
    PRESSURE_LIMIT_MAX,
    PRESSURE_LIMIT_MIN,
    PRESSURE_SLIDER_MAX,
    PRESSURE_SLIDER_MIN,
)

APP_UI_PATH = Path(REPO_ROOT) / "src" / "app_ui.py"


@pytest.fixture()
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_predict_rejects_physically_impossible_pressure(client):
    assert (
        client.post("/predict", json={"pressure": -1.0, "current": 15.0}).status_code
        == 422
    )
    assert (
        client.post("/predict", json={"pressure": 999.0, "current": 15.0}).status_code
        == 422
    )


def test_predict_rejects_physically_impossible_current(client):
    assert (
        client.post("/predict", json={"pressure": 2.1, "current": -5.0}).status_code
        == 422
    )
    assert (
        client.post("/predict", json={"pressure": 2.1, "current": 999.0}).status_code
        == 422
    )


def test_predict_accepts_closed_interval_pressure_bounds(client):
    assert (
        client.post("/predict", json={"pressure": 0.0, "current": 15.0}).status_code
        == 200
    )
    assert (
        client.post("/predict", json={"pressure": 10.0, "current": 15.0}).status_code
        == 200
    )


def test_predict_accepts_closed_interval_current_bounds(client):
    assert (
        client.post("/predict", json={"pressure": 2.1, "current": 0.0}).status_code
        == 200
    )
    assert (
        client.post("/predict", json={"pressure": 2.1, "current": 50.0}).status_code
        == 200
    )


def test_health_reports_loaded_model_and_threshold(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "threshold": main.epsilon,
    }


def test_health_stays_ok_when_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(main, "mu", None)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_does_not_leak_model_path(client):
    response = client.get("/health")

    assert "models" not in response.text
    assert ".pkl" not in response.text


def _field_bounds(model, name):
    ge = next(
        rule.ge
        for rule in model.model_fields[name].metadata
        if isinstance(rule, annotated_types.Ge)
    )
    le = next(
        rule.le
        for rule in model.model_fields[name].metadata
        if isinstance(rule, annotated_types.Le)
    )
    return ge, le


def test_sensor_data_fields_use_shared_limits():
    assert _field_bounds(main.SensorData, "pressure") == (
        PRESSURE_LIMIT_MIN,
        PRESSURE_LIMIT_MAX,
    )
    assert _field_bounds(main.SensorData, "current") == (
        CURRENT_LIMIT_MIN,
        CURRENT_LIMIT_MAX,
    )


def test_validation_limits_cover_ui_slider_ranges():
    assert PRESSURE_LIMIT_MIN <= PRESSURE_SLIDER_MIN
    assert PRESSURE_LIMIT_MAX >= PRESSURE_SLIDER_MAX
    assert CURRENT_LIMIT_MIN <= CURRENT_SLIDER_MIN
    assert CURRENT_LIMIT_MAX >= CURRENT_SLIDER_MAX


def test_app_ui_sliders_import_shared_slider_limits():
    source = APP_UI_PATH.read_text(encoding="utf-8")

    for name in (
        "PRESSURE_SLIDER_MIN",
        "PRESSURE_SLIDER_MAX",
        "CURRENT_SLIDER_MIN",
        "CURRENT_SLIDER_MAX",
    ):
        assert name in source
