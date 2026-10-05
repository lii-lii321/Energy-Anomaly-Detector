import importlib
import os
import sys

import pytest

from conftest import REPO_ROOT


def test_train_rejects_model_path_outside_models_dir(monkeypatch):
    monkeypatch.setenv("ENERGY_MODEL_PATH", os.path.join(REPO_ROOT, "evil.pkl"))
    sys.modules.pop("train", None)
    try:
        with pytest.raises(ValueError):
            importlib.import_module("train")
    finally:
        sys.modules.pop("train", None)
