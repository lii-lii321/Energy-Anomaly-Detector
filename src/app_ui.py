import streamlit as st

from api_client import DEFAULT_BASE_URL, ApiUnavailable, call_api
from sensor_limits import (
    CURRENT_SLIDER_MAX,
    CURRENT_SLIDER_MIN,
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
