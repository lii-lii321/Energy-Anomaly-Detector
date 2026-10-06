# 🛢️ 能源设备多维运行状态监测系统 (Energy-Anomaly-Detector)

![CI](https://github.com/lii-lii321/Energy-Anomaly-Detector/actions/workflows/ci.yml/badge.svg)

> **基于多元高斯分布 (Multivariate Gaussian) 的工业级异常检测方案**

## 📖 项目背景
针对石油/能源行业中单一阈值报警无法识别“逻辑冲突型”异常（如：压力降低但电流激增）的痛点，本项目开发了一套基于统计学习的实时监测系统。该系统能捕捉传感器数据的多维相关性，实现毫秒级风险预警。

## 🌟 核心亮点 (Key Features)
- **跨维度逻辑检测**：利用 **协方差矩阵 (Covariance Matrix)** 建模特征耦合关系，精准识别违反物理逻辑的隐蔽故障。
- **阈值自动寻优**：训练时在标注样本（200 条正常 + 1 条注入异常）上基于 **F1-Score** 自动搜索最优概率阈值 ε（`src/threshold.py` 的 `find_best_threshold`），并把 ε 作为模型元数据写入 `models/energy_model.pkl`；推理服务优先读取元数据中的 ε，旧模型文件自动回退默认 1e-5。
- **全栈工程架构**：
  - **算法层**：Python + SciPy 实现多元高斯建模与模型持久化。
  - **服务层**：基于 **FastAPI** 封装异步推理接口，支持高并发数据流处理。
  - **展示层**：使用 **Streamlit** 构建可视化驾驶舱，提供 3D 概率曲面分析与实时报警。
  - **工程护栏**：模型输出路径守卫（`src/model_paths.py`，越出 `models/` 目录的 `ENERGY_MODEL_PATH` 会被拒绝）+ pytest 回归测试。

## 🛠️ 技术栈 (Tech Stack)
- **Core**: Python 3.9+, NumPy, Pandas, SciPy
- **Web**: FastAPI, Uvicorn, Streamlit
- **Visualization**: Matplotlib (3D), Streamlit Charts
- **Testing**: pytest, FastAPI TestClient

## 📂 项目结构
```text
Energy-Anomaly-Detector/
├── src/                        # 核心代码
│   ├── train.py                # 模型训练与 3D 可视化
│   ├── main.py                 # FastAPI 后端推理服务
│   ├── sensor_limits.py        # 传感器物理量程常量（API 校验与 UI 滑条共用）
│   ├── api_client.py           # /predict 客户端：超时、状态码检查、友好错误
│   ├── app_ui.py               # Streamlit 前端交互界面
│   ├── model_paths.py          # 模型输出路径守卫
│   └── threshold.py            # F1 阈值搜索工具函数
├── tests/                      # pytest 测试套件
├── scripts/                    # 辅助脚本
│   └── real_time_monitor.py    # 实时监测命令行演示
├── models/                     # 训练好的模型权重 (pkl)
├── assets/                     # 演示截图
├── .github/                    # GitHub Actions CI 配置
├── requirements.txt            # 项目依赖
└── .gitignore
```

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python src/train.py
python src/train.py --show
```
脚本会把多元高斯参数 μ/σ 与 F1 自动选出的阈值 ε 一起保存到 `models/energy_model.pkl`（另含 `n_samples`、`threshold_method`、`trained_at` 元数据）。默认以无显示的 Agg 后端运行：把 3D 概率曲面（蓝色为正常点、红色为异常点）保存到 `assets/3d_plot.png` 后自然退出，可在 Linux 服务器 / CI 等无显示环境复现仓库演示图；只有加 `--show` 才会在保存后弹出交互窗口。μ/σ 仅用 200 条正常样本估计，注入异常仅参与阈值评估，避免离群点污染参数估计。可通过环境变量 `ENERGY_MODEL_PATH` 指定 `models/` 目录内的其他输出路径，越出该目录的路径会被拒绝。

### 3. 启动推理服务
```bash
uvicorn src.main:app --port 8000
```

### 4. 调用推理接口
服务启动后，另开一个终端执行：
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"pressure\":1.7,\"current\":18.0}"
```
真实响应示例：
```json
{
  "status": "success",
  "prediction": {
    "is_anomaly": true,
    "probability": 5.212464536070528e-18,
    "threshold": 0.0026430229932921964
  },
  "message": "检测到异常运行"
}
```

服务还提供 `GET /health` 探活接口：返回 `status`、`model_loaded`、`threshold` 三个字段，供部署探活与演示排障使用（不暴露模型文件路径）。

`/predict` 对入参做物理范围校验（闭区间）：井口压力 0–10 MPa、电机电流 0–50 A，负压力、超标电流等物理上不可能的读数会被直接拒绝（422），不会进入概率计算；校验边界严格覆盖前端滑条量程（压力 1.0–3.0 MPa、电流 10.0–25.0 A，两组量程统一维护在 `src/sensor_limits.py`），拖动滑条演示永远不会触发 422。

### 5. 可视化驾驶舱
保持推理服务运行，另开一个终端执行：
```bash
streamlit run src/app_ui.py
```
驾驶舱默认请求 `http://127.0.0.1:8000`，如后端换了端口或主机，可在页面左侧边栏的"后端地址"输入框直接修改（也可在启动前设置环境变量 `ENERGY_API_BASE`）。请求带 3 秒超时并检查状态码：后端未启动或返回非 2xx 时，点击"开始诊断"不会抛出原始堆栈，而是给出中文提示（含 `uvicorn src.main:app --port 8000` 启动命令）。

### 6. 运行测试
本地跑完整测试套件：
```bash
python -m pytest -q
```
推送后 GitHub Actions（`.github/workflows/ci.yml`）会在 `ubuntu-latest` 上执行同样的 `pip install -r requirements.txt` + `python -m pytest -q` 门禁，全绿才算通过。

## 📊 系统演示 (System Demo)

### 1. 3D 概率曲面分析
下图展示了多元高斯模型对正常数据（蓝色点）与异常数据（红色点）的区分能力。正常数据聚集在高概率密度“山峰”区域，而异常点落在低概率平原。
![3D图](assets/3d_plot.png)
### 2. 实时监测仪表盘
基于 Streamlit 构建的交互式前端，当检测到“压力降低且电流激增”的逻辑冲突时，系统会立即触发红色警报。
![仪表盘](assets/dashboard_alert.png)
