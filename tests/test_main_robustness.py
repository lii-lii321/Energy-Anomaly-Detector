import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO_ROOT
from fastapi.testclient import TestClient
from scipy.stats import multivariate_normal

import main
from model_paths import DEFAULT_MODEL_PATH


@pytest.fixture()
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_predict_never_leaks_internal_error_to_client(client, caplog, monkeypatch):
    def broken_pdf(*args, **kwargs):
        raise Exception("SECRET-INTERNAL")

    monkeypatch.setattr(multivariate_normal, "pdf", broken_pdf)

    with caplog.at_level(logging.ERROR):
        response = client.post("/predict", json={"pressure": 2.1, "current": 15.0})

    assert response.status_code == 500
    assert "SECRET-INTERNAL" not in response.text
    assert response.json()["detail"] == "推理失败，请查看服务端日志"
    assert any(
        "SECRET-INTERNAL" in record.getMessage() for record in caplog.records
    )


def test_model_path_reuses_model_paths_constant():
    assert main.MODEL_PATH == DEFAULT_MODEL_PATH

    source = (Path(REPO_ROOT) / "src" / "main.py").read_text(encoding="utf-8")
    assert "../models" not in source


def test_model_loads_in_lifespan_not_at_import(monkeypatch):
    monkeypatch.setattr(main, "mu", None)
    monkeypatch.setattr(main, "sigma", None)

    bare_client = TestClient(main.app)
    assert main.mu is None
    response = bare_client.post("/predict", json={"pressure": 2.1, "current": 15.0})
    assert response.status_code == 500
    assert "模型未加载" in response.json()["detail"]

    with TestClient(main.app):
        assert isinstance(main.mu, np.ndarray)
        assert main.sigma.shape == (2, 2)


def test_uvicorn_startup_path_remains_importable():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from uvicorn.importer import import_from_string; "
            "import_from_string('src.main:app'); print('ok')",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
