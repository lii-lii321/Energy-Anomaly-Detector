import logging
import pickle
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import multivariate_normal

try:
    from model_paths import DEFAULT_MODEL_PATH
    from sensor_limits import (
        CURRENT_LIMIT_MAX,
        CURRENT_LIMIT_MIN,
        PRESSURE_LIMIT_MAX,
        PRESSURE_LIMIT_MIN,
    )
except ImportError:  # uvicorn 以 src.main:app 启动时 src/ 不在 sys.path
    from src.model_paths import DEFAULT_MODEL_PATH
    from src.sensor_limits import (
        CURRENT_LIMIT_MAX,
        CURRENT_LIMIT_MIN,
        PRESSURE_LIMIT_MAX,
        PRESSURE_LIMIT_MIN,
    )

logger = logging.getLogger(__name__)

MODEL_PATH = DEFAULT_MODEL_PATH
DEFAULT_EPSILON = 1e-5

mu = None
sigma = None
epsilon = DEFAULT_EPSILON


def validate_model_params(model_params):
    if not isinstance(model_params, dict):
        raise ValueError("模型参数必须是 dict")
    try:
        mu_arr = np.asarray(model_params["mu"], dtype=float)
        sigma_arr = np.asarray(model_params["sigma"], dtype=float)
    except KeyError as e:
        raise ValueError(f"模型参数缺少字段：{e.args[0]}") from e
    except (TypeError, ValueError) as e:
        raise ValueError("mu/sigma 必须是数值数组") from e

    if mu_arr.ndim != 1 or mu_arr.shape[0] != 2:
        raise ValueError("mu 必须为一维且长度为 2")
    if sigma_arr.shape != (2, 2):
        raise ValueError("sigma 必须为 2x2 矩阵")
    if not np.allclose(sigma_arr, sigma_arr.T):
        raise ValueError("sigma 必须为对称正定矩阵")
    try:
        np.linalg.cholesky(sigma_arr)
    except (np.linalg.LinAlgError, ValueError) as e:
        raise ValueError("sigma 必须为对称正定矩阵") from e

    epsilon_value = model_params.get("epsilon", DEFAULT_EPSILON)
    try:
        epsilon_value = float(epsilon_value)
    except (TypeError, ValueError) as e:
        raise ValueError("epsilon 必须为正数") from e
    if not epsilon_value > 0:
        raise ValueError("epsilon 必须为正数")
    return mu_arr, sigma_arr, epsilon_value


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
    try:
        mu, sigma, epsilon = validate_model_params(model_params)
    except ValueError as e:
        logger.error("模型参数校验失败：%s", e)
        mu = None
        sigma = None
        epsilon = DEFAULT_EPSILON
        return
    logger.info("成功加载模型：%s", MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="能源设备异常监测系统 API", lifespan=lifespan)


class SensorData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"pressure": 2.1, "current": 15.0}],
        }
    )

    pressure: float = Field(
        ge=PRESSURE_LIMIT_MIN,
        le=PRESSURE_LIMIT_MAX,
        description="井口压力 (MPa)",
    )
    current: float = Field(
        ge=CURRENT_LIMIT_MIN,
        le=CURRENT_LIMIT_MAX,
        description="电机电流 (A)",
    )


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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": mu is not None,
        "threshold": epsilon,
    }


@app.get("/")
async def root():
    return {"message": "Energy Monitoring System is Running"}
