import numpy as np
import pytest

from threshold import compute_f1, compute_prf, find_best_threshold


def test_compute_prf_known_precision_and_recall():
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 1, 0, 0]

    metrics = compute_prf(y_true, y_pred)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_compute_prf_zero_division_branches_score_zero():
    assert compute_prf([0, 0, 1, 1], [0, 0, 0, 0]) == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert compute_prf([0, 0], [1, 1]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_compute_prf_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compute_prf([], [])
    with pytest.raises(ValueError):
        compute_prf([1, 0], [1])
    with pytest.raises(ValueError):
        compute_prf([1, 2], [1, 0])
    with pytest.raises(ValueError):
        compute_prf([1, 0], [1, 2])


def test_compute_f1_matches_compute_prf_f1():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 1, 0]

    assert compute_f1(y_true, y_pred) == compute_prf(y_true, y_pred)["f1"]


def test_compute_f1_known_precision_and_recall():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 1, 0]

    precision = 2 / 3
    recall = 2 / 3
    expected = 2 * precision * recall / (precision + recall)

    assert compute_f1(y_true, y_pred) == pytest.approx(expected)


def test_compute_f1_is_zero_without_true_positives():
    assert compute_f1([0, 0], [1, 0]) == 0.0
    assert compute_f1([1, 1], [0, 0]) == 0.0


def test_find_best_threshold_separates_perfectly_on_hand_computed_case():
    y_true = [0, 0, 0, 1]
    probs = [0.9, 0.8, 0.5, 0.1]

    threshold, f1 = find_best_threshold(y_true, probs)

    assert threshold == 0.5
    assert f1 == 1.0
    assert list((np.asarray(probs) < threshold).astype(int)) == y_true


def test_find_best_threshold_recovers_planted_anomaly_in_imbalanced_data():
    rng = np.random.default_rng(0)
    normal_probs = rng.uniform(0.5, 1.0, size=200)
    anomaly_probs = np.array([0.01, 0.02, 0.03])
    probs = np.concatenate([anomaly_probs, normal_probs])
    y_true = np.array([1] * 3 + [0] * 200)

    threshold, f1 = find_best_threshold(y_true, probs)

    assert f1 == 1.0
    assert 0.03 < threshold <= 1.0
    assert list((probs < threshold).astype(int)) == list(y_true)


def test_find_best_threshold_reports_zero_f1_when_labels_are_all_negative():
    threshold, f1 = find_best_threshold([0, 0, 0], [0.7, 0.4, 0.1])

    assert f1 == 0.0
    assert threshold == 0.1


def test_find_best_threshold_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        find_best_threshold([1, 0], [0.5])
    with pytest.raises(ValueError):
        find_best_threshold([], [])
    with pytest.raises(ValueError):
        find_best_threshold([1, 2], [0.5, 0.4])
