import argparse
import pickle

import numpy as np
from scipy.stats import multivariate_normal

from model_paths import DEFAULT_MODEL_PATH
from threshold import compute_prf
from train import NORMAL_COV, NORMAL_MEAN

HOLDOUT_SEED = 123
DEFAULT_N_NORMAL = 200
DEFAULT_N_ANOMALY = 20
ANOMALY_SIGMA_LOW = 2.1
ANOMALY_SIGMA_HIGH = 3.0


def compute_metrics(y_true, y_pred):
    """兼容入口：precision/recall/F1 统一实现在 threshold.compute_prf。"""
    return compute_prf(y_true, y_pred)


def build_holdout_samples(
    n_normal=DEFAULT_N_NORMAL, n_anomaly=DEFAULT_N_ANOMALY, seed=HOLDOUT_SEED
):
    for name, count in (("n_normal", n_normal), ("n_anomaly", n_anomaly)):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rng = np.random.default_rng(seed)
    normal = rng.multivariate_normal(NORMAL_MEAN, NORMAL_COV, n_normal)
    sigma_p = np.sqrt(NORMAL_COV[0][0])
    sigma_c = np.sqrt(NORMAL_COV[1][1])
    severity = rng.uniform(ANOMALY_SIGMA_LOW, ANOMALY_SIGMA_HIGH, n_anomaly)
    anomalies = np.column_stack(
        [
            NORMAL_MEAN[0] - severity * sigma_p,
            NORMAL_MEAN[1] + severity * sigma_c,
        ]
    )
    features = np.vstack([normal, anomalies])
    labels = np.array([0] * n_normal + [1] * n_anomaly)
    return features, labels


def evaluate(model_data, n_normal=DEFAULT_N_NORMAL, n_anomaly=DEFAULT_N_ANOMALY, seed=HOLDOUT_SEED):
    if not isinstance(model_data, dict):
        raise ValueError("model_data must be a dict containing mu/sigma/epsilon")
    try:
        mu = np.asarray(model_data["mu"], dtype=float)
        sigma = np.asarray(model_data["sigma"], dtype=float)
        epsilon = float(model_data["epsilon"])
    except KeyError as e:
        raise ValueError(f"model_data is missing field: {e.args[0]}") from e
    except (TypeError, ValueError) as e:
        raise ValueError("mu/sigma must be numeric arrays and epsilon must be numeric") from e
    if mu.ndim != 1 or sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("mu must be 1-D and sigma must be a square matrix")
    if not epsilon > 0:
        raise ValueError("epsilon must be positive")

    features, y_true = build_holdout_samples(
        n_normal=n_normal, n_anomaly=n_anomaly, seed=seed
    )
    probabilities = multivariate_normal.pdf(features, mean=mu, cov=sigma)
    y_pred = (probabilities < epsilon).astype(int)
    metrics = compute_metrics(y_true, y_pred)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "epsilon": epsilon,
    }


def load_committed_model(path=DEFAULT_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def format_metrics(result):
    return (
        f"precision={result['precision']:.4f} recall={result['recall']:.4f} "
        f"f1={result['f1']:.4f} "
        f"(n_normal={result['n_normal']}, n_anomaly={result['n_anomaly']}, "
        f"epsilon={result['epsilon']:.6g})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在与训练集不同种子的留出集上评估模型")
    parser.add_argument("--n-normal", type=int, default=DEFAULT_N_NORMAL)
    parser.add_argument("--n-anomaly", type=int, default=DEFAULT_N_ANOMALY)
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    model = load_committed_model(args.model_path or None)
    result = evaluate(model, args.n_normal, args.n_anomaly, args.seed)
    print(format_metrics(result))
