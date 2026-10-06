import os

import requests

DEFAULT_BASE_URL = os.environ.get("ENERGY_API_BASE", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT = 3.0
_START_COMMAND = "uvicorn src.main:app --port 8000"
_ALLOWED_SCHEMES = ("http://", "https://")


class ApiUnavailable(Exception):
    """后端 /predict 不可达、返回非 2xx 或响应格式异常时的友好错误。"""


def call_api(pressure, current, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url.startswith(_ALLOWED_SCHEMES):
        raise ApiUnavailable(
            f"后端地址必须以 http:// 或 https:// 开头，当前为：{base_url or '(空)'}"
        )
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"pressure": pressure, "current": current},
            timeout=timeout,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise ApiUnavailable(
            f"无法连接后端服务（{base_url}），请先在另一个终端执行：{_START_COMMAND}"
        ) from exc
    if not 200 <= response.status_code < 300:
        raise ApiUnavailable(
            f"后端返回异常状态码 {response.status_code}，请确认推理服务正常运行：{_START_COMMAND}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiUnavailable(
            "后端响应不是有效 JSON，请确认地址指向本项目的推理服务"
        ) from exc
    if not isinstance(payload, dict) or "prediction" not in payload:
        raise ApiUnavailable(
            "后端响应缺少 prediction 字段，请确认地址指向本项目的 /predict 接口"
        )
    return payload
