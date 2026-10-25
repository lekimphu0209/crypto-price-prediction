"""
System Status page - trained models from the registry (real) + service hints.

The model table is real (GET /api/models); the placeholder uptime/latency
tiles are marked DEMO since there is no monitoring backend yet.
"""
import pandas as pd
import streamlit as st

from components.badges import demo_badge, real_badge
from services.api_client import list_models


def render():
    st.title("⚙ System Status")
    st.markdown("---")

    demo_badge("Các chỉ số uptime/response/samples bên dưới là placeholder demo.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Sources", "2/2 Online", "✅")
    with col2:
        st.metric("API Response", "125ms", "✅")
    with col3:
        st.metric("Training Samples", "-", "")

    st.subheader("Trained Models")
    models = list_models()
    if not models:
        st.info("API offline hoặc chưa có model nào được train.")
        return
    real_badge("Danh sách model thật từ models/*/meta.json")

    rows = []
    for m in models:
        metrics = m.get("metrics") or {}
        rows.append({
            "Symbol": m.get("symbol"),
            "Model": m.get("name"),
            "Balanced Acc": metrics.get("balanced_accuracy"),
            "Accuracy": metrics.get("accuracy"),
            "F1": metrics.get("f1"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
