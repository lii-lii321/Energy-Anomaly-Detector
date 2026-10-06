import json
from pathlib import Path

import pytest
import requests
from conftest import REPO_ROOT

import api_client
from api_client import DEFAULT_TIMEOUT, ApiUnavailable, call_api

APP_UI_PATH = Path(REPO_ROOT) / "src" / "app_ui.py"


def _response(status_code, payload):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


def test_call_api_posts_json_payload_and_returns_parsed_prediction(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, json=json, timeout=timeout)
        return _response(
            200,
            {
                "status": "success",
                "prediction": {"is_anomaly": False, "probability": 0.5},
            },
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = call_api(2.1, 15.0, base_url="http://127.0.0.1:8000", timeout=1.5)

    assert captured["url"] == "http://127.0.0.1:8000/predict"
    assert captured["json"] == {"pressure": 2.1, "current": 15.0}
    assert captured["timeout"] == 1.5
    assert isinstance(result, dict)
    assert "prediction" in result


def test_call_api_uses_default_base_url_and_timeout(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, timeout=timeout)
        return _response(200, {"prediction": {"is_anomaly": False}})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = call_api(2.1, 15.0)

    assert captured["url"] == f"{api_client.DEFAULT_BASE_URL}/predict"
    assert captured["timeout"] == DEFAULT_TIMEOUT
    assert result == {"prediction": {"is_anomaly": False}}


def test_call_api_wraps_timeout_in_friendly_error(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api(2.1, 15.0)

    message = str(excinfo.value)
    assert "后端" in message
    assert "uvicorn src.main:app --port 8000" in message


def test_call_api_wraps_connection_error_in_friendly_error(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api(2.1, 15.0)

    message = str(excinfo.value)
    assert "无法连接后端服务" in message
    assert "uvicorn" in message


def test_call_api_wraps_http_500_in_friendly_error(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda url, json=None, timeout=None: _response(500, {"detail": "模型未加载"}),
    )

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api(2.1, 15.0)

    assert excinfo.type is ApiUnavailable
    assert "500" in str(excinfo.value)
    assert "uvicorn" in str(excinfo.value)


def test_call_api_wraps_non_json_body_in_friendly_error(monkeypatch):
    response = requests.Response()
    response.status_code = 200
    response._content = b"<html>not json</html>"
    monkeypatch.setattr(
        api_client.requests, "post", lambda url, json=None, timeout=None: response
    )

    with pytest.raises(ApiUnavailable):
        call_api(2.1, 15.0)


def test_call_api_rejects_payload_without_prediction_key(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda url, json=None, timeout=None: _response(200, {"unexpected": True}),
    )

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api(2.1, 15.0)

    assert "prediction" in str(excinfo.value)


def test_call_api_rejects_non_http_base_url_without_sending(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise AssertionError("非法 scheme 不应发起网络请求")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api(2.1, 15.0, base_url="ftp://127.0.0.1:8000")

    assert "http" in str(excinfo.value)


def test_app_ui_has_no_hardcoded_address_or_raw_response_access():
    source = APP_UI_PATH.read_text(encoding="utf-8")

    assert "127.0.0.1" not in source
    assert "response.json()" not in source
    assert "from api_client import" in source
    assert "ApiUnavailable" in source


def test_app_ui_batch_panel_uses_client_and_shared_limits():
    source = APP_UI_PATH.read_text(encoding="utf-8")

    assert "call_api_batch" in source
    assert "file_uploader" in source
    assert "download_button" in source
    assert "PRESSURE_LIMIT_MIN" in source
    assert "PRESSURE_LIMIT_MAX" in source
    assert "CURRENT_LIMIT_MIN" in source
    assert "CURRENT_LIMIT_MAX" in source
    assert "st.error" in source


def test_api_client_stays_a_pure_logic_module():
    source = (Path(REPO_ROOT) / "src" / "api_client.py").read_text(encoding="utf-8")

    assert "import streamlit" not in source
    assert "requests" in source
