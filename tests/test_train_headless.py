import importlib
import os
import subprocess
import sys

import matplotlib
from scipy.stats import multivariate_normal

matplotlib.use("Agg")

from conftest import REPO_ROOT

from threshold import find_best_threshold
from train import (
    build_labeled_samples,
    fit_normal_parameters,
    plot_3d_anomaly,
    score_labeled_samples,
)

TRAIN_SCRIPT = os.path.join(REPO_ROOT, "src", "train.py")
COMMITTED_PLOT_PATH = os.path.join(REPO_ROOT, "assets", "3d_plot.png")
TMP_MODEL_PATH = os.path.join(REPO_ROOT, "models", "_tmp_train.pkl")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _labeled_frame():
    features, labels = build_labeled_samples()
    mu, sigma = fit_normal_parameters(features, labels)
    probabilities = multivariate_normal.pdf(features, mean=mu, cov=sigma)
    epsilon, _ = find_best_threshold(labels, probabilities)
    model_data = {"mu": mu, "sigma": sigma, "epsilon": epsilon}
    return score_labeled_samples(model_data), mu, sigma


def test_plot_with_output_path_saves_png_without_show(tmp_path):
    df, mu, sigma = _labeled_frame()
    output = tmp_path / "p.png"

    returned = plot_3d_anomaly(df, mu, sigma, output_path=str(output))

    assert returned == str(output)
    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.read_bytes()[: len(PNG_MAGIC)] == PNG_MAGIC


def test_train_source_keeps_qt_backend_out_of_module_level():
    with open(TRAIN_SCRIPT, encoding="utf-8") as f:
        source = f.read()
    module_level, main_block = source.split('if __name__ == "__main__":', 1)

    assert "Qt5Agg" not in module_level
    assert 'matplotlib.use("Agg")' in main_block


def test_importing_train_does_not_change_matplotlib_backend():
    backend_before = matplotlib.get_backend()
    sys.modules.pop("train", None)
    try:
        importlib.import_module("train")
    finally:
        sys.modules.pop("train", None)

    backend_after = matplotlib.get_backend()
    assert backend_after == backend_before
    assert "qt" not in backend_after.lower()


def test_train_script_runs_headless_by_default():
    env = dict(os.environ, ENERGY_MODEL_PATH=TMP_MODEL_PATH)
    mtime_before = os.path.getmtime(COMMITTED_PLOT_PATH)
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert os.path.isfile(TMP_MODEL_PATH)
        assert os.path.getmtime(COMMITTED_PLOT_PATH) > mtime_before
    finally:
        if os.path.exists(TMP_MODEL_PATH):
            os.remove(TMP_MODEL_PATH)
