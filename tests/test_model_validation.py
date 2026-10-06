import logging
import pickle
from pathlib import Path

import numpy as np
import pytest

import main
from conftest import REPO_ROOT
from model_paths import DEFAULT_MODEL_PATH

VALID_MU = [2.1, 15.0]
VALID_SIGMA = [[0.01, 0.008], [0.008, 0.1]]


def _write_model(tmp_path, payload):
    path = tmp_path / "energy_model.pkl"
    path.write_bytes(pickle.dumps(payload))
    return str(path)


def _load_with(monkeypatch, caplog, model_path):
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "mu", None)
    monkeypatch.setattr(main, "sigma", None)
    monkeypatch.setattr(main, "epsilon", 0.123)
    with caplog.at_level(logging.ERROR):
        main.load_model()


def _trained_mu_sigma():
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        trained = pickle.load(f)
    return trained["mu"], trained["sigma"]


def test_non_positive_definite_sigma_rejected_at_load(tmp_path, monkeypatch, caplog):
    path = _write_model(
        tmp_path, {"mu": VALID_MU, "sigma": [[1, 2], [2, 1]], "epsilon": 0.01}
    )

    _load_with(monkeypatch, caplog, path)

    assert main.mu is None
    assert main.sigma is None
    assert any("校验失败" in record.getMessage() for record in caplog.records)


def test_singular_sigma_rejected_at_load(tmp_path, monkeypatch, caplog):
    path = _write_model(
        tmp_path, {"mu": VALID_MU, "sigma": [[1, 1], [1, 1]], "epsilon": 0.01}
    )

    _load_with(monkeypatch, caplog, path)

    assert main.mu is None
    assert main.sigma is None
    assert any("校验失败" in record.getMessage() for record in caplog.records)


def test_wrong_dimension_mu_rejected(tmp_path, monkeypatch, caplog):
    path = _write_model(
        tmp_path, {"mu": [1.0, 2.0, 3.0], "sigma": np.eye(3), "epsilon": 0.01}
    )

    _load_with(monkeypatch, caplog, path)

    assert main.mu is None
    assert any("校验失败" in record.getMessage() for record in caplog.records)


def test_valid_model_loads_with_explicit_epsilon(tmp_path, monkeypatch, caplog):
    mu_trained, sigma_trained = _trained_mu_sigma()
    path = _write_model(
        tmp_path,
        {"mu": mu_trained, "sigma": sigma_trained, "epsilon": 0.0025},
    )

    _load_with(monkeypatch, caplog, path)

    assert isinstance(main.mu, np.ndarray)
    assert main.mu.shape == (2,)
    assert main.sigma.shape == (2, 2)
    assert main.epsilon == pytest.approx(0.0025)


def test_valid_model_loads_with_default_epsilon_fallback(tmp_path, monkeypatch, caplog):
    mu_trained, sigma_trained = _trained_mu_sigma()
    path = _write_model(tmp_path, {"mu": mu_trained, "sigma": sigma_trained})

    _load_with(monkeypatch, caplog, path)

    assert isinstance(main.mu, np.ndarray)
    assert main.epsilon == pytest.approx(main.DEFAULT_EPSILON)


def test_real_model_pkl_passes_validation():
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        trained = pickle.load(f)

    mu_arr, sigma_arr, epsilon_value = main.validate_model_params(trained)

    assert mu_arr.shape == (2,)
    assert sigma_arr.shape == (2, 2)
    assert epsilon_value > 0
    np.linalg.cholesky(sigma_arr)


def test_validate_rejects_non_dict_payload():
    with pytest.raises(ValueError):
        main.validate_model_params([VALID_MU, VALID_SIGMA])


def test_validate_rejects_non_positive_epsilon(tmp_path, monkeypatch, caplog):
    path = _write_model(
        tmp_path, {"mu": VALID_MU, "sigma": VALID_SIGMA, "epsilon": 0.0}
    )

    _load_with(monkeypatch, caplog, path)

    assert main.mu is None
    assert any("校验失败" in record.getMessage() for record in caplog.records)


def test_real_model_still_loads_via_load_model(monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", str(DEFAULT_MODEL_PATH))
    monkeypatch.setattr(main, "mu", None)
    monkeypatch.setattr(main, "sigma", None)

    main.load_model()

    assert isinstance(main.mu, np.ndarray)
    assert main.sigma.shape == (2, 2)
    assert main.epsilon > 0


def test_readme_mentions_startup_sigma_validation():
    readme = (Path(REPO_ROOT) / "README.md").read_text(encoding="utf-8")
    assert "对称正定" in readme
