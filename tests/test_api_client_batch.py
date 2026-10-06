import json

import pytest
import requests

import api_client
from api_client import ApiUnavailable, call_api_batch


def _response(status_code, payload):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


def test_call_api_batch_posts_sample_list_and_returns_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, json=json, timeout=timeout)
        return _response(
            200,
            {
                "status": "success",
                "n_total": 2,
                "n_anomaly": 1,
                "predictions": [
                    {"pressure": 2.1, "current": 15.0, "is_anomaly": False},
                    {"pressure": 1.7, "current": 18.5, "is_anomaly": True},
                ],
            },
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = call_api_batch(
        [(2.1, 15.0), (1.7, 18.5)], base_url="http://127.0.0.1:8000", timeout=1.5
    )

    assert captured["url"] == "http://127.0.0.1:8000/predict/batch"
    assert captured["json"] == {
        "samples": [{"pressure": 2.1, "current": 15.0}, {"pressure": 1.7, "current": 18.5}]
    }
    assert captured["timeout"] == 1.5
    assert result["n_total"] == 2
    assert len(result["predictions"]) == 2


def test_call_api_batch_wraps_timeout_in_friendly_error(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api_batch([(2.1, 15.0)])

    message = str(excinfo.value)
    assert "无法连接后端服务" in message
    assert "uvicorn src.main:app --port 8000" in message


def test_call_api_batch_wraps_http_500_in_friendly_error(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda url, json=None, timeout=None: _response(500, {"detail": "模型未加载"}),
    )

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api_batch([(2.1, 15.0)])

    assert "500" in str(excinfo.value)


def test_call_api_batch_wraps_non_json_body_in_friendly_error(monkeypatch):
    response = requests.Response()
    response.status_code = 200
    response._content = b"<html>not json</html>"
    monkeypatch.setattr(
        api_client.requests, "post", lambda url, json=None, timeout=None: response
    )

    with pytest.raises(ApiUnavailable):
        call_api_batch([(2.1, 15.0)])


def test_call_api_batch_rejects_payload_without_predictions_key(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda url, json=None, timeout=None: _response(200, {"unexpected": True}),
    )

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api_batch([(2.1, 15.0)])

    assert "predictions" in str(excinfo.value)


def test_call_api_batch_rejects_non_list_predictions(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda url, json=None, timeout=None: _response(200, {"predictions": "oops"}),
    )

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api_batch([(2.1, 15.0)])

    assert "predictions" in str(excinfo.value)


def test_call_api_batch_rejects_non_http_base_url_without_sending(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise AssertionError("非法 scheme 不应发起网络请求")

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    with pytest.raises(ApiUnavailable) as excinfo:
        call_api_batch([(2.1, 15.0)], base_url="ftp://127.0.0.1:8000")

    assert "http" in str(excinfo.value)


def test_call_api_batch_accepts_empty_sample_list_and_posts_min_payload(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json=json)
        return _response(200, {"predictions": []})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = call_api_batch([])

    assert captured["json"] == {"samples": []}
    assert result == {"predictions": []}
