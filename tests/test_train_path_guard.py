import os

import pytest

from conftest import REPO_ROOT
from train import train


def test_train_rejects_model_path_outside_models_dir(monkeypatch):
    monkeypatch.setenv("ENERGY_MODEL_PATH", os.path.join(REPO_ROOT, "evil.pkl"))
    with pytest.raises(ValueError):
        train()


def test_train_rejects_explicit_model_path_outside_models_dir():
    with pytest.raises(ValueError):
        train(os.path.join(REPO_ROOT, "evil.pkl"))
