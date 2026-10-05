import logging
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import multivariate_normal

try:
    from model_paths import DEFAULT_MODEL_PATH
except ImportError:  # uvicorn 以 src.main:app 启动时 src/ 不在 sys.path
    from src.model_paths import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)

MODEL_PATH = DEFAULT_MODEL_PATH

mu = None
sigma = None
epsilon = 1e-5


def load_model():
    global mu, sigma, epsilon
    try:
        with open(MODEL_PATH, "rb") as f:
            model_params = pickle.load(f)
    except FileNotFoundError:
        logger.warning(
            "未找到模型文件：%s，请先运行 src/train.py 生成模型", MODEL_PATH
        )
        return
    mu = model_params["mu"]
    sigma = model_params["sigma"]
    epsilon = model_params.get("epsilon", 1e-5)
    logger.info("成功加载模型：%s", MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="能源设备异常监测系统 API", lifespan=lifespan)


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
        logger.exception(e)
        raise HTTPException(status_code=500, detail="推理失败，请查看服务端日志") from e


@app.get("/")
async def root():
    return {"message": "Energy Monitoring System is Running"}
