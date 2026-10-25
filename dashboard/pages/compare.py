"""
Model Comparison page - real metrics from the trained-model registry.

Reads GET /api/models (models/{SYMBOL}/{slug}/meta.json) and renders a
metrics table + bar charts. No invented numbers: missing metrics show as
gaps instead of fake values.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.badges import real_badge
from config import SYMBOLS
from services.api_client import list_models

_METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "r2", "mae"]
_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Acc",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "r2": "R²",
    "mae": "MAE",
}


def _table(models: list, symbol: str) -> pd.DataFrame:
    rows = []
    for m in models:
        if m.get("symbol") != symbol:
            continue
        metrics = m.get("metrics") or {}
        row = {"Model": m.get("name", m.get("type"))}
        for k in _METRIC_KEYS:
            row[_METRIC_LABELS[k]] = metrics.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def render():
    st.title("🤖 Model Comparison")
    st.markdown("---")

    models = list_models()
    if not models:
        st.error("Không lấy được danh sách model từ API. Kiểm tra server backend đã chạy chưa.")
        return
    real_badge("Metrics thật từ models/*/meta.json qua backend API")

    symbol = st.selectbox("Symbol", SYMBOLS, index=0)
    df = _table(models, symbol)
    if df.empty:
        st.info(f"Chưa có model nào được train cho {symbol}.")
        return

    st.dataframe(df, use_container_width=True)

    chart_keys = [k for k in ["balanced_accuracy", "accuracy", "f1", "r2"] if df[_METRIC_LABELS[k]].notna().any()]
    if not chart_keys:
        return
    cols = st.columns(len(chart_keys))
    for col, k in zip(cols, chart_keys):
        label = _METRIC_LABELS[k]
        with col:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["Model"], y=df[label]))
            fig.update_layout(title=label, height=300)
            st.plotly_chart(fig, use_container_width=True)
