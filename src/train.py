import argparse
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from model_paths import (
    ASSETS_DIR,
    DEFAULT_PLOT_PATH,
    MODELS_DIR,
    resolve_model_path,
)
from threshold import find_best_threshold

# 支持通过环境变量 ENERGY_MODEL_PATH 指定 models/ 目录内的输出文件名，越界路径会被拒绝
RANDOM_SEED = 42
NORMAL_SAMPLES = 200
NORMAL_MEAN = [2.1, 15.0]
NORMAL_COV = [[0.01, 0.008], [0.008, 0.1]]
INJECTED_ANOMALY = [1.7, 18.0]
FEATURE_COLUMNS = ["pressure", "current"]


def build_labeled_samples():
    np.random.seed(RANDOM_SEED)
    normal_data = np.random.multivariate_normal(
        NORMAL_MEAN, NORMAL_COV, NORMAL_SAMPLES
    )
    features = np.vstack([normal_data, np.array([INJECTED_ANOMALY])])
    labels = np.array([0] * NORMAL_SAMPLES + [1])
    return features, labels


def fit_normal_parameters(features, labels):
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features must be 2-D and labels must match its row count")
    normal = features[labels == 0]
    if normal.shape[0] < 2:
        raise ValueError("at least two normal samples are required to fit covariance")
    return normal.mean(axis=0), np.cov(normal, rowvar=False)


def train(model_path=None):
    requested = (
        model_path if model_path is not None else os.environ.get("ENERGY_MODEL_PATH")
    )
    resolved = Path(resolve_model_path(requested)).resolve()
    filename = resolved.name
    if ".." in filename or os.sep in filename:
        raise ValueError(f"invalid model file name: {filename}")
    models_root = Path(MODELS_DIR).resolve()
    model_path = models_root / filename
    if model_path.parent != models_root:
        raise ValueError(f"model path must stay inside {MODELS_DIR}")

    features, labels = build_labeled_samples()
    mu_2d, sigma_2d = fit_normal_parameters(features, labels)

    probabilities = multivariate_normal.pdf(features, mean=mu_2d, cov=sigma_2d)
    epsilon, best_f1 = find_best_threshold(labels, probabilities)

    model_data = {
        "mu": mu_2d,
        "sigma": sigma_2d,
        "epsilon": epsilon,
        "n_samples": len(labels),
        "threshold_method": "f1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(pickle.dumps(model_data))
    print(f"✅ 模型已成功保存至: {model_path}")
    print(f"✅ F1 自动选阈值: epsilon={epsilon:.6g}, f1={best_f1:.3f}, n={len(labels)}")
    return model_data


def score_labeled_samples(model_data):
    features, _ = build_labeled_samples()
    df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    df["probability"] = multivariate_normal.pdf(
        features, mean=model_data["mu"], cov=model_data["sigma"]
    )
    df["is_anomaly"] = df["probability"] < model_data["epsilon"]
    return df


def plot_3d_anomaly(df, mu, sigma, output_path=None):
    if output_path is not None and not str(output_path).strip():
        raise ValueError("output_path must be a non-empty string when provided")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    x = np.linspace(df["pressure"].min() - 0.1, df["pressure"].max() + 0.1, 100)
    y = np.linspace(df["current"].min() - 1, df["current"].max() + 1, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, sigma)
    Z = rv.pdf(pos)
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)
    normal = df[df["is_anomaly"] == False]
    ax.scatter(
        normal["pressure"], normal["current"], normal["probability"], c="blue", s=20
    )
    anomaly = df[df["is_anomaly"] == True]
    ax.scatter(
        anomaly["pressure"],
        anomaly["current"],
        anomaly["probability"],
        c="red",
        s=100,
        marker="x",
    )
    ax.set_xlabel("Pressure (MPa)")
    ax.set_ylabel("Current (A)")
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练能源设备异常检测模型")
    parser.add_argument(
        "--show",
        action="store_true",
        help="训练并保存 assets/3d_plot.png 后弹出 3D 交互窗口（默认无头运行，不弹窗）",
    )
    args = parser.parse_args()

    if args.show:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            pass
    else:
        matplotlib.use("Agg")

    trained = train()
    os.makedirs(ASSETS_DIR, exist_ok=True)
    df = score_labeled_samples(trained)
    plot_3d_anomaly(df, trained["mu"], trained["sigma"], output_path=DEFAULT_PLOT_PATH)
    print(f"✅ 3D 概率曲面已保存至: {DEFAULT_PLOT_PATH}")
    if args.show:
        plot_3d_anomaly(df, trained["mu"], trained["sigma"])
