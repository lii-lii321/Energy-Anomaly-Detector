import os
import pickle
import sys

import numpy as np
import pytest
from conftest import REPO_ROOT
from scipy.stats import multivariate_normal

SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import real_time_monitor as monitor  # noqa: E402

VALID_MU = [2.1, 15.0]
VALID_SIGMA = [[0.01, 0.008], [0.008, 0.1]]


def _write_model(tmp_path, payload):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(payload))
    return str(path)


def test_load_monitor_model_reads_epsilon_from_metadata(tmp_path):
    path = _write_model(
        tmp_path, {"mu": VALID_MU, "sigma": VALID_SIGMA, "epsilon": 0.0025}
    )

    model = monitor.load_monitor_model(path)

    assert model["epsilon"] == pytest.approx(0.0025)
    assert not np.allclose(model["epsilon"], monitor.DEFAULT_EPSILON)


def test_load_monitor_model_falls_back_to_default_epsilon(tmp_path):
    path = _write_model(tmp_path, {"mu": VALID_MU, "sigma": VALID_SIGMA})

    model = monitor.load_monitor_model(path)

    assert model["epsilon"] == pytest.approx(monitor.DEFAULT_EPSILON)


def test_load_monitor_model_rejects_missing_fields(tmp_path):
    path = _write_model(tmp_path, {"mu": VALID_MU})

    with pytest.raises(ValueError):
        monitor.load_monitor_model(path)


def test_load_monitor_model_rejects_non_positive_epsilon(tmp_path):
    path = _write_model(
        tmp_path, {"mu": VALID_MU, "sigma": VALID_SIGMA, "epsilon": 0.0}
    )

    with pytest.raises(ValueError):
        monitor.load_monitor_model(path)


def test_check_anomaly_uses_metadata_epsilon_not_hardcoded_default():
    # 概率约 5.9e-3：高于默认 1e-5 但低于元数据 ε=0.01，
    # 若硬编码旧默认值会被误判为正常（回归守卫）
    model = {"mu": [0.0, 0.0], "sigma": [[1e-4, 0.0], [0.0, 1e-4]], "epsilon": 0.01}

    is_anomaly, prob = monitor.check_anomaly(model, 0.05, 0.0)

    expected = multivariate_normal.pdf(
        [0.05, 0.0], mean=model["mu"], cov=model["sigma"]
    )
    assert prob == pytest.approx(expected, rel=1e-12)
    assert prob > monitor.DEFAULT_EPSILON
    assert is_anomaly is True


def test_check_anomaly_reports_normal_point_as_normal():
    model = {"mu": [0.0, 0.0], "sigma": [[1e-4, 0.0], [0.0, 1e-4]], "epsilon": 0.01}

    is_anomaly, _ = monitor.check_anomaly(model, 0.0, 0.0)

    assert is_anomaly is False


def test_classify_stream_flags_injected_anomaly_with_committed_model():
    model = monitor.load_monitor_model()

    results = monitor.classify_stream(model)

    assert [r["is_anomaly"] for r in results] == [False, False, True]
    assert results[2]["pressure"] == 1.70
    assert results[2]["current"] == 18.5


def test_classify_stream_is_empty_for_empty_stream():
    model = monitor.load_monitor_model()

    assert monitor.classify_stream(model, stream=[]) == []


def test_main_prints_metadata_threshold_and_stream_verdicts(tmp_path, capsys):
    path = _write_model(
        tmp_path, {"mu": VALID_MU, "sigma": VALID_SIGMA, "epsilon": 0.0025}
    )

    monitor.main(["--model-path", path])

    output = capsys.readouterr().out
    assert "ε=0.0025" in output
    assert "异常报警" in output
    assert "运行正常" in output


def test_format_result_marks_anomaly_and_normal_differently():
    anomaly = {"pressure": 1.7, "current": 18.5, "is_anomaly": True, "probability": 1e-18}
    normal = {"pressure": 2.05, "current": 14.8, "is_anomaly": False, "probability": 0.2}

    assert "异常报警" in monitor.format_result(anomaly)
    assert "运行正常" in monitor.format_result(normal)
