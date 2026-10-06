"""命令行演示：加载模型后对模拟的实时数据流逐点判定。

阈值 ε 优先读取模型元数据（训练时由 F1 自动寻优写入），
元数据缺失时回退 DEFAULT_EPSILON，与推理服务 /predict 的语义保持一致。
"""

import argparse
import os
import pickle
import sys

import numpy as np
from scipy.stats import multivariate_normal

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model_paths import DEFAULT_MODEL_PATH  # noqa: E402

DEFAULT_EPSILON = 1e-5

# 模拟三条新产生的数据：正常、正常、严重异常
DEMO_STREAM = [
    (2.05, 14.8),  # 正常波动
    (2.15, 15.2),  # 正常波动
    (1.70, 18.5),  # 模拟漏油：压力骤降且电流激增
]


def load_monitor_model(path=DEFAULT_MODEL_PATH):
    with open(path, "rb") as f:
        params = pickle.load(f)
    if not isinstance(params, dict):
        raise ValueError("模型文件内容必须是 dict")
    try:
        mu = np.asarray(params["mu"], dtype=float)
        sigma = np.asarray(params["sigma"], dtype=float)
    except KeyError as e:
        raise ValueError(f"模型文件缺少字段：{e.args[0]}") from e
    epsilon = float(params.get("epsilon", DEFAULT_EPSILON))
    if not epsilon > 0:
        raise ValueError("epsilon 必须为正数")
    return {"mu": mu, "sigma": sigma, "epsilon": epsilon}


def check_anomaly(model, pressure, current):
    point = np.array([pressure, current])
    prob = float(multivariate_normal.pdf(point, mean=model["mu"], cov=model["sigma"]))
    return prob < model["epsilon"], prob


def classify_stream(model, stream=DEMO_STREAM):
    results = []
    for pressure, current in stream:
        is_anomaly, prob = check_anomaly(model, pressure, current)
        results.append(
            {
                "pressure": pressure,
                "current": current,
                "is_anomaly": is_anomaly,
                "probability": prob,
            }
        )
    return results


def format_result(result):
    status = "⚠️ [异常报警]" if result["is_anomaly"] else "✅ [运行正常]"
    return (
        f"当前工况: 压力={result['pressure']}MPa, 电流={result['current']}A | "
        f"判定: {status} (概率: {result['probability']:.2e})"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="智慧油井实时监测命令行演示")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args(argv)

    model = load_monitor_model(args.model_path)
    print("--- 智慧油井实时监测系统已启动 ---")
    print(f"使用模型元数据阈值 ε={model['epsilon']:.6g}")
    for result in classify_stream(model):
        print(format_result(result))


if __name__ == "__main__":
    main()
