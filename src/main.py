import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from scipy.stats import multivariate_normal

app = FastAPI(title="能源设备异常监测系统 API")

# --- 核心修复：自动获取绝对路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../models/energy_model.pkl")

# 初始化变量
mu = None
sigma = None
epsilon = 1e-5

# 加载模型
try:
    with open(model_path, "rb") as f:
        model_params = pickle.load(f)
    mu = model_params["mu"]
    sigma = model_params["sigma"]
    print(f"✅ 成功加载模型：{os.path.abspath(model_path)}")
except FileNotFoundError:
    print(
        f"❌ 错误：找不到模型文件。程序尝试查找的路径是：{os.path.abspath(model_path)}"
    )
    print("请先运行 src/train.py 生成模型！")


class SensorData(BaseModel):
    pressure: float
    current: float


@app.post("/predict")
async def predict_status(data: SensorData):
    if mu is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务器日志")

    try:
        sample = np.array([data.pressure, data.current])
        prob = multivariate_normal.pdf(sample, mean=mu, cov=sigma)
        is_anomaly = bool(prob < epsilon)
        return {
            "status": "success",
            "prediction": {
                "is_anomaly": is_anomaly,
                "probability": float(prob),
                "threshold": epsilon,
            },
            "message": "检测到异常运行" if is_anomaly else "系统运行正常",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Energy Monitoring System is Running"}
