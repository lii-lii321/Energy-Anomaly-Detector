import os

import pytest
from conftest import REPO_ROOT

from model_paths import DEFAULT_MODEL_PATH, MODELS_DIR, resolve_model_path


def test_default_path_points_at_the_trained_model():
    assert DEFAULT_MODEL_PATH == os.path.join(REPO_ROOT, "models", "energy_model.pkl")
    assert resolve_model_path() == DEFAULT_MODEL_PATH


def test_path_inside_models_dir_is_resolved():
    resolved = resolve_model_path(os.path.join(MODELS_DIR, "energy_model_v2.pkl"))

    assert resolved == os.path.join(MODELS_DIR, "energy_model_v2.pkl")


@pytest.mark.parametrize(
    "escaping_path",
    [
        os.path.join(MODELS_DIR, "..", "energy_model.pkl"),
        os.path.join(MODELS_DIR, "..", "..", "evil.pkl"),
        os.path.join(REPO_ROOT, "evil.pkl"),
        "C:\\Windows\\evil.pkl",
    ],
)
def test_paths_leaving_the_models_dir_are_rejected(escaping_path):
    with pytest.raises(ValueError):
        resolve_model_path(escaping_path)


def test_empty_path_is_rejected():
    with pytest.raises(ValueError):
        resolve_model_path("   ")
