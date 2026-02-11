import pickle
import numpy as np
from scipy.stats import multivariate_normal

# 1. 加载我们训练好的“专家经验”（模型参数）
with open("energy_model.pkl", "rb") as f:
    model_params = pickle.load(f)

mu = model_params["mu"]
sigma = model_params["sigma"]
epsilon = 1e-5  # 这是我们之前选出的最优阈值


def check_anomaly(pressure, current):
    """
    输入实时监测到的压力和电流，返回是否异常
    """
    # 构造当前数据点
    current_point = np.array([pressure, current])

    # 使用多元高斯公式计算概率
    prob = multivariate_normal.pdf(current_point, mean=mu, cov=sigma)

    if prob < epsilon:
        return True, prob
    return False, prob


# 2. 模拟实时数据流
print("--- 智慧油井实时监测系统已启动 ---")

# 模拟三条新产生的数据：正常、正常、严重异常
test_data = [
    (2.05, 14.8),  # 正常波动
    (2.15, 15.2),  # 正常波动
    (1.70, 18.5),  # 模拟漏油：压力骤降且电流激增
]

for p, c in test_data:
    is_anomaly, p_val = check_anomaly(p, c)
    status = "⚠️ [异常报警]" if is_anomaly else "✅ [运行正常]"
    print(f"当前工况: 压力={p}MPa, 电流={c}A | 判定: {status} (概率: {p_val:.2e})")
