import os

import requests

DEFAULT_BASE_URL = os.environ.get("ENERGY_API_BASE", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT = 3.0
_START_COMMAND = "uvicorn src.main:app --port 8000"
_ALLOWED_SCHEMES = ("http://", "https://")


class ApiUnavailable(Exception):
    """后端 /predict 不可达、返回非 2xx 或响应格式异常时的友好错误。"""


def _require_http_base_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url.startswith(_ALLOWED_SCHEMES):
        raise ApiUnavailable(
            f"后端地址必须以 http:// 或 https:// 开头，当前为：{base_url or '(空)'}"
        )
    return base_url


def _post_json(base_url, path, payload, timeout):
    try:
        return requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise ApiUnavailable(
            f"无法连接后端服务（{base_url}），请先在另一个终端执行：{_START_COMMAND}"
        ) from exc


def _parse_success_payload(response, required_key, endpoint):
    if not 200 <= response.status_code < 300:
        raise ApiUnavailable(
            f"后端返回异常状态码 {response.status_code}，"
            f"请确认推理服务正常运行：{_START_COMMAND}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiUnavailable(
            "后端响应不是有效 JSON，请确认地址指向本项目的推理服务"
        ) from exc
    if not isinstance(payload, dict) or required_key not in payload:
        raise ApiUnavailable(
            f"后端响应缺少 {required_key} 字段，请确认地址指向本项目的 {endpoint} 接口"
        )
    return payload


def call_api(pressure, current, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT):
    base_url = _require_http_base_url(base_url)
    response = _post_json(
        base_url,
        "/predict",
        {"pressure": pressure, "current": current},
        timeout,
    )
    return _parse_success_payload(response, "prediction", "/predict")


def call_api_batch(samples, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT):
    """批量调用 /predict/batch。samples 为 (pressure, current) 二元组序列。"""
    base_url = _require_http_base_url(base_url)
    payload_samples = [{"pressure": p, "current": c} for p, c in samples]
    response = _post_json(
        base_url,
        "/predict/batch",
        {"samples": payload_samples},
        timeout,
    )
    payload = _parse_success_payload(response, "predictions", "/predict/batch")
    if not isinstance(payload["predictions"], list):
        raise ApiUnavailable(
            "后端响应的 predictions 字段不是列表，请确认地址指向本项目的 /predict/batch 接口"
        )
    return payload
