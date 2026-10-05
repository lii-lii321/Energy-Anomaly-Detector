import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "models"))
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "energy_model.pkl")
ASSETS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "assets"))
DEFAULT_PLOT_PATH = os.path.join(ASSETS_DIR, "3d_plot.png")


def is_inside_models_dir(path):
    candidate = os.path.abspath(path)
    if candidate == MODELS_DIR:
        return False
    try:
        return os.path.commonpath([candidate, MODELS_DIR]) == MODELS_DIR
    except ValueError:
        return False


def resolve_model_path(user_path=None):
    if user_path is None:
        return DEFAULT_MODEL_PATH
    if not isinstance(user_path, str) or not user_path.strip():
        raise ValueError("model path must be a non-empty string")
    candidate = os.path.abspath(user_path)
    if not is_inside_models_dir(candidate):
        raise ValueError(f"model path must stay inside {MODELS_DIR}")
    return candidate
