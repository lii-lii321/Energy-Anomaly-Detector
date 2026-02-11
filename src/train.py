import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib

# --- 关键修改：启用交互式后端 ---
try:
    matplotlib.use("Qt5Agg")
except:
    pass

# --- 核心修复：自动获取绝对路径，不再依赖运行位置 ---
# 获取当前脚本所在目录 (.../src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算出 models 文件夹的绝对路径 (.../models/energy_model.pkl)
model_path = os.path.join(current_dir, "../models/energy_model.pkl")

# 1. 模拟数据
np.random.seed(42)
n_samples = 200
mean_val = [2.1, 15.0]
cov_val = [[0.01, 0.008], [0.008, 0.1]]
normal_data = np.random.multivariate_normal(mean_val, cov_val, n_samples)
anomaly_data = np.array([[1.7, 18.0]])
all_data = np.vstack([normal_data, anomaly_data])
df_2d = pd.DataFrame(all_data, columns=["pressure", "current"])

# 2. 计算参数
mu_2d = df_2d.mean().values
sigma_2d = np.cov(df_2d.values, rowvar=False)

# 3. 保存模型 (使用修复后的路径)
model_data = {"mu": mu_2d, "sigma": sigma_2d}

# 确保父目录存在
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(model_data, f)
print(f"✅ 模型已成功保存至: {os.path.abspath(model_path)}")

# 4. 绘图 (这部分保持不变)
p_val_2d = multivariate_normal.pdf(df_2d.values, mean=mu_2d, cov=sigma_2d)
df_2d["probability"] = p_val_2d
epsilon = 1e-5
df_2d["is_anomaly"] = df_2d["probability"] < epsilon


def plot_3d_anomaly(df, mu, sigma):
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
    plt.show()


# 如果直接运行此脚本，则绘图
if __name__ == "__main__":
    plot_3d_anomaly(df_2d, mu_2d, sigma_2d)
