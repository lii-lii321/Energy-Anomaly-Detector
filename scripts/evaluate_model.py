import argparse
import os
import sys

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from evaluate import (  # noqa: E402
    DEFAULT_N_ANOMALY,
    DEFAULT_N_NORMAL,
    HOLDOUT_SEED,
    evaluate,
    format_metrics,
    load_committed_model,
)
from model_paths import resolve_model_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="在留出集（与训练集不同 seed）上评估 models/ 下的已提交模型"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="models/ 目录内的模型文件路径，默认 models/energy_model.pkl",
    )
    parser.add_argument("--n-normal", type=int, default=DEFAULT_N_NORMAL)
    parser.add_argument("--n-anomaly", type=int, default=DEFAULT_N_ANOMALY)
    parser.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    args = parser.parse_args()

    model_data = load_committed_model(resolve_model_path(args.model_path))
    print(format_metrics(evaluate(model_data, args.n_normal, args.n_anomaly, args.seed)))


if __name__ == "__main__":
    main()
