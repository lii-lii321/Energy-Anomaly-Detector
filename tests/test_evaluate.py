import os
import pickle
import re

import numpy as np
import pytest
from conftest import REPO_ROOT

from evaluate import (
    ANOMALY_SIGMA_HIGH,
    ANOMALY_SIGMA_LOW,
    build_holdout_samples,
    compute_metrics,
    evaluate,
)
from train import NORMAL_COV, NORMAL_MEAN, build_labeled_samples

COMMITTED_MODEL_PATH = os.path.join(REPO_ROOT, "models", "energy_model.pkl")
README_PATH = os.path.join(REPO_ROOT, "README.md")
HOLDOUT_TABLE_ROW = re.compile(r"^\|.*留出集.*$", re.MULTILINE)
FLOAT_NUMBER = re.compile(r"\d+\.\d+")


def _load_committed_model():
    with open(COMMITTED_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_compute_metrics_matches_hand_computed_counts():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 1, 0, 0]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_compute_metrics_all_predicted_normal_scores_zero():
    metrics = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0])

    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_compute_metrics_all_predicted_anomaly_degrades_precision_only():
    metrics = compute_metrics([0, 0, 1, 1], [1, 1, 1, 1])

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_compute_metrics_perfect_predictions_score_one():
    y_true = [0, 0, 1, 1]

    assert compute_metrics(y_true, y_true) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_compute_metrics_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [1])
    with pytest.raises(ValueError):
        compute_metrics([1, 2], [1, 0])
    with pytest.raises(ValueError):
        compute_metrics([1, 0], [1, 2])


def test_holdout_generation_is_reproducible():
    features_a, labels_a = build_holdout_samples()
    features_b, labels_b = build_holdout_samples()

    assert np.array_equal(features_a, features_b)
    assert np.array_equal(labels_a, labels_b)


def test_holdout_samples_follow_distribution_and_injection_spec():
    n_normal, n_anomaly = 50, 5
    features, labels = build_holdout_samples(
        n_normal=n_normal, n_anomaly=n_anomaly, seed=7
    )

    assert features.shape == (n_normal + n_anomaly, 2)
    assert list(labels[:n_normal]) == [0] * n_normal
    assert list(labels[n_normal:]) == [1] * n_anomaly

    sigma_c = np.sqrt(NORMAL_COV[1][1])
    severity = (features[n_normal:, 1] - NORMAL_MEAN[1]) / sigma_c
    assert np.all(severity >= ANOMALY_SIGMA_LOW)
    assert np.all(severity <= ANOMALY_SIGMA_HIGH)
    assert np.all(
        features[n_normal:, 0] < NORMAL_MEAN[0]
    ), "anomalies must sit on the pressure-drop / current-rise conflict direction"


def test_holdout_uses_a_different_seed_than_training():
    holdout_features, _ = build_holdout_samples()
    train_features, _ = build_labeled_samples()

    assert not np.array_equal(holdout_features[:200], train_features[:200])


def test_evaluate_is_reproducible_for_fixed_seed():
    model = _load_committed_model()

    assert evaluate(model) == evaluate(model)


def test_evaluate_returns_all_reported_fields():
    result = evaluate(_load_committed_model(), n_normal=100, n_anomaly=10, seed=5)

    assert set(result) == {
        "precision",
        "recall",
        "f1",
        "n_normal",
        "n_anomaly",
        "epsilon",
    }
    assert result["n_normal"] == 100
    assert result["n_anomaly"] == 10
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1


def test_committed_model_clears_f1_floor_on_holdout():
    result = evaluate(_load_committed_model())

    assert result["f1"] >= 0.8
    assert result["epsilon"] > 0


def test_evaluate_rejects_invalid_model_data():
    with pytest.raises(ValueError):
        evaluate({"mu": [2.1, 15.0], "sigma": [[0.01, 0], [0, 0.1]]})
    with pytest.raises(ValueError):
        evaluate({"mu": [2.1, 15.0], "sigma": [[0.01], [0, 0.1]], "epsilon": 1e-5})
    with pytest.raises(ValueError):
        evaluate(
            {"mu": [2.1, 15.0], "sigma": [[0.01, 0], [0, 0.1]], "epsilon": -1.0}
        )
    with pytest.raises(ValueError):
        evaluate("not-a-dict")


def test_readme_evaluation_table_matches_holdout_output():
    result = evaluate(_load_committed_model())
    readme = open(README_PATH, encoding="utf-8").read()

    assert "模型评估" in readme, "README must document the holdout evaluation"
    rows = [
        line for line in readme.splitlines() if line.startswith("|") and "留出集" in line
    ]
    assert len(rows) == 1, "README evaluation table must have exactly one holdout row"

    numbers = [float(token) for token in FLOAT_NUMBER.findall(rows[0])]
    assert len(numbers) == 4, f"holdout row must list epsilon/precision/recall/f1: {rows[0]}"
    epsilon, precision, recall, f1 = numbers
    assert epsilon == pytest.approx(result["epsilon"], abs=1e-6)
    assert precision == pytest.approx(result["precision"], abs=1e-4)
    assert recall == pytest.approx(result["recall"], abs=1e-4)
    assert f1 == pytest.approx(result["f1"], abs=1e-4)
