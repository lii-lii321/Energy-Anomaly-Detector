import importlib
import os
import pickle
import sys

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from conftest import REPO_ROOT
from threshold import find_best_threshold
from train import build_labeled_samples, fit_normal_parameters, train

COMMITTED_MODEL_PATH = os.path.join(REPO_ROOT, "models", "energy_model.pkl")
TMP_MODEL_PATH = os.path.join(REPO_ROOT, "models", "energy_model_pipeline_tmp.pkl")


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_train_persists_f1_selected_epsilon(monkeypatch):
    monkeypatch.setenv("ENERGY_MODEL_PATH", TMP_MODEL_PATH)
    if os.path.exists(TMP_MODEL_PATH):
        os.remove(TMP_MODEL_PATH)
    try:
        train()
        model = _load(TMP_MODEL_PATH)
    finally:
        if os.path.exists(TMP_MODEL_PATH):
            os.remove(TMP_MODEL_PATH)

    assert 0 < model["epsilon"] < 1
    assert model["threshold_method"] == "f1"
    assert model["n_samples"] == 201

    features, labels = build_labeled_samples()
    probs = multivariate_normal.pdf(features, mean=model["mu"], cov=model["sigma"])
    threshold, f1 = find_best_threshold(labels, probs)
    assert f1 == 1.0
    assert model["epsilon"] == pytest.approx(threshold)


def test_fit_normal_parameters_uses_only_normal_rows():
    normal = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    features = np.vstack([normal, [[100.0, -50.0]]])
    labels = np.array([0, 0, 0, 0, 0, 1])

    mu, sigma = fit_normal_parameters(features, labels)

    assert mu == pytest.approx(normal.mean(axis=0))
    assert np.allclose(sigma, np.cov(normal, rowvar=False))
    assert not np.allclose(mu, features.mean(axis=0))
    assert not np.allclose(sigma, np.cov(features, rowvar=False))


def test_trained_model_parameters_come_from_normal_samples_only(monkeypatch):
    monkeypatch.setenv("ENERGY_MODEL_PATH", TMP_MODEL_PATH)
    if os.path.exists(TMP_MODEL_PATH):
        os.remove(TMP_MODEL_PATH)
    try:
        train()
        model = _load(TMP_MODEL_PATH)
    finally:
        if os.path.exists(TMP_MODEL_PATH):
            os.remove(TMP_MODEL_PATH)

    features, labels = build_labeled_samples()
    normal = features[labels == 0]
    assert model["mu"] == pytest.approx(normal.mean(axis=0))
    assert np.allclose(model["sigma"], np.cov(normal, rowvar=False))


def test_importing_train_module_does_not_retrain_the_model():
    mtime_before = os.path.getmtime(COMMITTED_MODEL_PATH)
    sys.modules.pop("train", None)
    try:
        importlib.import_module("train")
    finally:
        sys.modules.pop("train", None)

    assert os.path.getmtime(COMMITTED_MODEL_PATH) == mtime_before


def test_committed_model_artifact_carries_epsilon_metadata():
    model = _load(COMMITTED_MODEL_PATH)

    assert "epsilon" in model
    assert 0 < model["epsilon"] < 1
    assert model["threshold_method"] == "f1"
