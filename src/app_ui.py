import pandas as pd
import streamlit as st

from api_client import DEFAULT_BASE_URL, ApiUnavailable, call_api, call_api_batch
from sensor_limits import (
    CURRENT_LIMIT_MAX,
    CURRENT_LIMIT_MIN,
    CURRENT_SLIDER_MAX,
    CURRENT_SLIDER_MIN,
    PRESSURE_LIMIT_MAX,
    PRESSURE_LIMIT_MIN,
    PRESSURE_SLIDER_MAX,
    PRESSURE_SLIDER_MIN,
)

st.set_page_config(page_title="智慧油井实时监测系统", layout="wide")

st.title("🛢️ 智慧油井生产状态实时监测平台")
st.markdown("---")

backend_url = st.sidebar.text_input("后端地址", value=DEFAULT_BASE_URL)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 传感器数据输入")
    p = st.slider(
        "井口压力 (MPa)", PRESSURE_SLIDER_MIN, PRESSURE_SLIDER_MAX, 2.1, step=0.1
    )
    c = st.slider(
        "电机电流 (A)", CURRENT_SLIDER_MIN, CURRENT_SLIDER_MAX, 15.0, step=0.1
    )

    if st.button("开始诊断", use_container_width=True):
        try:
            result = call_api(p, c, base_url=backend_url)
        except ApiUnavailable as exc:
            st.error(str(exc))
        else:
            with col2:
                st.subheader("🔍 诊断结果")
                is_anomaly = result["prediction"]["is_anomaly"]

                if is_anomaly:
                    st.error("严重警告：检测到运行异常！")
                    st.metric(
                        "异常判定", "⚠️ 存在风险", delta="-100%", delta_color="inverse"
                    )
                else:
                    st.success("系统运行状态：正常")
                    st.metric("正常判定", "✅ 运行稳定", delta="安全")

                st.write(f"判定概率值: `{result['prediction']['probability']:.2e}`")
                st.progress(
                    max(0.0, min(1.0, result["prediction"]["probability"] * 2)),
                    text="系统健康度评分",
                )

st.markdown("---")

with st.expander("📦 批量诊断（上传 CSV）"):
    st.caption(
        "CSV 需包含 pressure、current 两列；超出物理量程"
        "（压力 0–10 MPa、电流 0–50 A）的行会被跳过，不参与诊断。"
    )
    uploaded = st.file_uploader("上传传感器数据 CSV", type=["csv"])

    if uploaded is not None and st.button("批量诊断", use_container_width=True):
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"CSV 解析失败：{exc}")
        else:
            missing = {"pressure", "current"} - set(frame.columns)
            if missing:
                st.error(f"CSV 缺少必需列：{', '.join(sorted(missing))}")
            else:
                try:
                    frame["pressure"] = pd.to_numeric(frame["pressure"])
                    frame["current"] = pd.to_numeric(frame["current"])
                except (TypeError, ValueError):
                    st.error("pressure / current 列必须全部为数值")
                else:
                    in_range = frame["pressure"].between(
                        PRESSURE_LIMIT_MIN, PRESSURE_LIMIT_MAX
                    ) & frame["current"].between(
                        CURRENT_LIMIT_MIN, CURRENT_LIMIT_MAX
                    )
                    skipped = int((~in_range).sum())
                    valid = frame[in_range]
                    if valid.empty:
                        st.error(
                            "没有通过物理量程校验的数据行"
                            "（压力 0–10 MPa、电流 0–50 A）"
                        )
                    else:
                        try:
                            batch = call_api_batch(
                                list(
                                    zip(
                                        valid["pressure"].astype(float).tolist(),
                                        valid["current"].astype(float).tolist(),
                                    )
                                ),
                                base_url=backend_url,
                            )
                        except ApiUnavailable as exc:
                            st.error(str(exc))
                        else:
                            results = pd.DataFrame(batch["predictions"])
                            total = len(results)
                            anomaly_count = int(results["is_anomaly"].sum())
                            metric1, metric2, metric3 = st.columns(3)
                            metric1.metric("样本总数", total)
                            metric2.metric("异常样本", anomaly_count)
                            metric3.metric("异常占比", f"{anomaly_count / total:.1%}")
                            st.dataframe(results)
                            if skipped:
                                st.warning(f"已跳过 {skipped} 条超出物理量程的数据行")
                            st.download_button(
                                "下载诊断结果 CSV",
                                data=results.to_csv(index=False).encode("utf-8"),
                                file_name="batch_diagnosis_results.csv",
                                mime="text/csv",
                            )
