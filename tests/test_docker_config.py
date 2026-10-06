from pathlib import Path

import pytest
from conftest import REPO_ROOT

DOCKERFILE_PATH = Path(REPO_ROOT) / "Dockerfile"
COMPOSE_PATH = Path(REPO_ROOT) / "docker-compose.yml"
DOCKERIGNORE_PATH = Path(REPO_ROOT) / ".dockerignore"


def _load_compose():
    yaml = pytest.importorskip("yaml")
    assert COMPOSE_PATH.is_file(), f"missing compose file at {COMPOSE_PATH}"
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(compose, dict), "docker-compose.yml must parse to a mapping"
    return compose


def _service_env(service):
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(key): str(value) for key, value in env.items()}
    pairs = {}
    for entry in env:
        key, _, value = str(entry).partition("=")
        pairs[key] = value
    return pairs


def test_compose_defines_api_and_ui_services():
    services = _load_compose().get("services") or {}
    assert "api" in services, "docker-compose.yml must define the api service"
    assert "ui" in services, "docker-compose.yml must define the ui service"


def test_api_service_maps_port_8000_and_checks_health():
    api = _load_compose()["services"]["api"]
    ports = [str(port) for port in api.get("ports") or []]
    assert "8000:8000" in ports, "api service must expose 8000:8000"
    healthcheck = api.get("healthcheck") or {}
    command = " ".join(str(part) for part in healthcheck.get("test") or [])
    assert "/health" in command, "api healthcheck must probe the /health endpoint"


def test_ui_runs_streamlit_against_api_and_maps_port_8501():
    ui = _load_compose()["services"]["ui"]
    command = " ".join(str(part) for part in ui.get("command") or [])
    assert "streamlit" in command and "app_ui.py" in command, (
        "ui service must run streamlit with src/app_ui.py"
    )
    assert (
        _service_env(ui).get("ENERGY_API_BASE") == "http://api:8000"
    ), "ui service must point ENERGY_API_BASE at http://api:8000"
    ports = [str(port) for port in ui.get("ports") or []]
    assert "8501:8501" in ports, "ui service must expose 8501:8501"
    depends_on = ui.get("depends_on")
    if isinstance(depends_on, dict):
        assert "api" in depends_on
    else:
        assert "api" in (depends_on or []), "ui service must depend on api"


def test_dockerfile_builds_uvicorn_api():
    assert DOCKERFILE_PATH.is_file(), f"missing Dockerfile at {DOCKERFILE_PATH}"
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "python:3.10-slim" in content, "Dockerfile must start from python:3.10-slim"
    assert "uvicorn" in content and "src.main:app" in content, (
        "Dockerfile CMD must start uvicorn with src.main:app"
    )
    assert "COPY models/ models/" in content, (
        "Dockerfile must bake in models/ so lifespan can load the pkl"
    )


def test_dockerignore_keeps_models_out_of_exclusions():
    assert DOCKERIGNORE_PATH.is_file(), f"missing .dockerignore at {DOCKERIGNORE_PATH}"
    excluded = {
        line.strip().rstrip("/")
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "models" not in excluded, (
        ".dockerignore must not exclude models/ (the runtime needs the pkl)"
    )
    assert "src" not in excluded, ".dockerignore must not exclude src/"
