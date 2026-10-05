import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "models"))
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "energy_model.pkl")


def resolve_model_path(user_path=None):
    if user_path is None:
        return DEFAULT_MODEL_PATH
    if not isinstance(user_path, str) or not user_path.strip():
        raise ValueError("model path must be a non-empty string")
    candidate = os.path.abspath(user_path)
    try:
        inside = os.path.commonpath([candidate, MODELS_DIR]) == MODELS_DIR
    except ValueError:
        inside = False
    if not inside or candidate == MODELS_DIR:
        raise ValueError(f"model path must stay inside {MODELS_DIR}")
    return candidate
