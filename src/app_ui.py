import streamlit as st
import requests

st.set_page_config(page_title="智慧油井实时监测系统", layout="wide")

st.title("🛢️ 智慧油井生产状态实时监测平台")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 传感器数据输入")
    p = st.slider("井口压力 (MPa)", 1.0, 3.0, 2.1, step=0.1)
    c = st.slider("电机电流 (A)", 10.0, 25.0, 15.0, step=0.1)

    if st.button("开始诊断", use_container_width=True):
        # 调用你刚才写好的 FastAPI 接口
        response = requests.post(
            "http://127.0.0.1:8000/predict", json={"pressure": p, "current": c}
        )
        result = response.json()

        with col2:
            st.subheader("🔍 诊断结果")
            is_anomaly = result["prediction"]["is_anomaly"]

            if is_anomaly:
                st.error(f"严重警告：检测到运行异常！")
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
