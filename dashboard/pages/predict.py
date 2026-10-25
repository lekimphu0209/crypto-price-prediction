"""
Trend Prediction page - direction (TANG/GIAM) + probability per symbol.

Models are up/down trend classifiers: the page shows the predicted direction,
its probability, and the decision threshold instead of a fake price point.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.badges import real_badge
from components.model_selector import model_selector
from config import SYMBOL_COLORS, TIMEFRAME_DAYS, SYMBOLS
from services.api_client import predict


def _metric_delta(res: dict) -> str:
    """Delta text for the direction metric."""
    proba = res.get("probability")
    if proba is not None:
        return f"P(UP)={proba:.2f} · thr={res.get('threshold', 0.5):.2f}"
    ret = res.get("predicted_return")
    if ret is not None:
        return f"E[return]={ret * 100:+.2f}%"
    return ""


def render():
    st.title("📈 Trend Prediction")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        model = model_selector()
    with col_b:
        timeframe = st.selectbox("History shown", list(TIMEFRAME_DAYS), index=1, key="predict_tf")

    days = TIMEFRAME_DAYS.get(timeframe, 30)

    with st.spinner(f"Đang gọi backend API (model {model})..."):
        results = {sym: predict(sym, model, days) for sym in SYMBOLS}

    ok_any = any(r.get("status") == "success" for r in results.values())
    if ok_any:
        real_badge("Dự đoán xu hướng thật từ model đã train (backend API)")
    else:
        err = next(iter(results.values()), {})
        st.error(f"Lỗi API: {err.get('message', 'Unknown error')}")
        return

    # Direction metrics, one pair per symbol
    cols = st.columns(len(SYMBOLS))
    for col, sym in zip(cols, SYMBOLS):
        res = results[sym]
        label = sym.replace("USDT", "")
        with col:
            if res.get("status") != "success":
                st.metric(label, "N/A", "")
                st.caption(res.get("message", ""))
                continue
            direction = res.get("direction_label") or res.get("direction") or "N/A"
            icon = "🟢" if res.get("direction") == "UP" else "🔴"
            st.metric(
                f"{label} · {res.get('horizon', 7)}d trend",
                f"{icon} {direction}",
                _metric_delta(res),
                delta_color="normal" if res.get("direction") == "UP" else "inverse",
            )
            st.caption(f"Giá hiện tại: ${res.get('current_price', 0):,.2f}")

    # Training-quality transparency line
    parts = []
    for sym in SYMBOLS:
        res = results[sym]
        if res.get("status") == "success":
            tm = res.get("train_metrics") or {}
            bits = [f"{sym.replace('USDT', '')}"]
            if "balanced_accuracy" in tm:
                bits.append(f"BalAcc={tm['balanced_accuracy']:.3f}")
            if "accuracy" in tm:
                bits.append(f"Acc={tm['accuracy']:.3f}")
            if "r2" in tm:
                bits.append(f"R²={tm['r2']:.3f}")
            parts.append(" ".join(bits))
    if parts:
        st.caption("Chất lượng trên tập kiểm tra — " + " · ".join(parts))

    # History chart (classifiers do not emit a price path)
    st.subheader("Price History")
    fig = make_subplots(
        rows=1, cols=len(SYMBOLS),
        subplot_titles=tuple(s.replace("USDT", "") for s in SYMBOLS),
        horizontal_spacing=0.08,
    )
    for idx, sym in enumerate(SYMBOLS, start=1):
        res = results[sym]
        short = sym.replace("USDT", "")
        if res.get("status") != "success":
            continue
        fig.add_trace(
            go.Scatter(
                x=res["hist_dates"], y=res["hist_values"],
                mode="lines", name=f"{short} Close", showlegend=False,
                line=dict(color=SYMBOL_COLORS.get(short, "#1f77b4"), width=2),
            ),
            row=1, col=idx,
        )
        fig.update_xaxes(title_text="Date", row=1, col=idx)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=idx)
    fig.update_layout(height=350, hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
