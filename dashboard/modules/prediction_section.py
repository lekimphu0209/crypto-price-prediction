"""
Price Prediction Section Module
Displays model predictions and forecasts
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import concurrent.futures

from .data_providers import (
    get_real_prices,
    get_historical_data,
    get_data_providers,
    timeframe_to_days,
)
import requests

SYMBOL_COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
}

# API Configuration
API_BASE_URL = "http://localhost:8502"


@st.cache_data(ttl=300, show_spinner=False)
def call_predict_api(symbol: str, model_name: str, display_days: int, horizon: int = 7) -> dict:
    """
    Call the backend API for prediction (cached for 5 minutes)

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        model_name: Model name
        display_days: Number of historical days to display
        horizon: Forecast horizon

    Returns:
        Prediction result dict
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/predict/",
            json={
                "symbol": symbol,
                "model_name": model_name,
                "display_days": display_days,
                "horizon": horizon
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"ok": False, "message": "Không kết nối được với backend API. Kiểm tra xem API server đã chạy chưa."}
    except Exception as e:
        return {"ok": False, "message": f"Lỗi API: {str(e)}"}


def render_prediction_section(timeframe, model):
    """Render Price Prediction section"""
    st.title("📈 Price Prediction Dashboard")
    st.markdown("---")

    st.caption(f"Model: **{model}** · Timeframe: **{timeframe}**")

    binance, _ = get_data_providers()
    days = timeframe_to_days(timeframe)
    horizon = 7

    # Run prediction using API - sequential calls for BTC and ETH
    with st.spinner(f"Đang gọi backend API để chạy model {model}..."):
        btc = call_predict_api("BTCUSDT", model, days, horizon=horizon)
        eth = call_predict_api("ETHUSDT", model, days, horizon=horizon)

    if btc.get("status") == "success" or eth.get("status") == "success":
        st.success(f"Model **{model}** (từ backend API) — dự đoán giá thật cho BTC & ETH.")
    else:
        st.error(f"Lỗi API: {btc.get('message', eth.get('message', 'Unknown error'))}")

    # Metrics (real, per symbol)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if btc.get("status") == "success" and btc.get("current_price"):
            st.metric("BTC Current", f"${btc['current_price']:,.2f}", "")
        else:
            st.metric("BTC Current", "N/A", "")
    with col2:
        if btc.get("status") == "success" and btc.get("predicted_tomorrow"):
            st.metric(
                "BTC Predicted (tomorrow)",
                f"${btc['predicted_tomorrow']:,.2f}",
                f"{btc['change_pct']:+.2f}%",
                delta_color="normal" if btc["change_pct"] >= 0 else "inverse",
            )
        else:
            st.metric("BTC Predicted (tomorrow)", "N/A", "")
    with col3:
        if eth.get("status") == "success" and eth.get("current_price"):
            st.metric("ETH Current", f"${eth['current_price']:,.2f}", "")
        else:
            st.metric("ETH Current", "N/A", "")
    with col4:
        if eth.get("status") == "success" and eth.get("predicted_tomorrow"):
            st.metric(
                "ETH Predicted (tomorrow)",
                f"${eth['predicted_tomorrow']:,.2f}",
                f"{eth['change_pct']:+.2f}%",
                delta_color="normal" if eth["change_pct"] >= 0 else "inverse",
            )
        else:
            st.metric("ETH Predicted (tomorrow)", "N/A", "")

    # Confidence / quality (real, from held-out test set)
    if btc.get("status") == "success" or eth.get("status") == "success":
        parts = []
        if btc.get("status") == "success":
            parts.append(f"BTC R²={btc['train_r2']:.3f}, MAE=${btc['train_mae']:,.0f}")
        if eth.get("status") == "success":
            parts.append(f"ETH R²={eth['train_r2']:.3f}, MAE=${eth['train_mae']:,.0f}")
        st.caption("Độ chính xác trên tập kiểm tra — " + " · ".join(parts))

    # Price forecast chart - show both BTC and ETH side by side
    st.subheader("Price Forecast (real model)")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Bitcoin (BTC)", "Ethereum (ETH)"),
        horizontal_spacing=0.08,
    )

    for col_idx, (symbol, res) in enumerate([("BTC", btc), ("ETH", eth)], start=1):
        if res.get("status") != "success":
            fig.add_annotation(
                text=res.get("message", "N/A"),
                xref="x domain" if col_idx == 1 else "x2 domain",
                yref="y domain" if col_idx == 1 else "y2 domain",
                x=0.5, y=0.5, showarrow=False,
            )
            continue

        show_legend = col_idx == 1
        fig.add_trace(
            go.Scatter(
                x=res["hist_dates"],
                y=res["hist_values"],
                mode="lines",
                name="Historical",
                legendgroup="hist",
                showlegend=show_legend,
                line=dict(color=SYMBOL_COLORS.get(symbol, "#1f77b4"), width=2),
            ),
            row=1, col=col_idx,
        )
        # Connect last historical point to first forecast point
        connect_x = list(res["forecast_dates"])
        connect_y = list(res["forecast_values"])
        fig.add_trace(
            go.Scatter(
                x=[res["hist_dates"][-1]] + connect_x,
                y=[res["hist_values"][-1]] + connect_y,
                mode="lines+markers",
                name=f"Predicted {horizon} Days",
                legendgroup="pred",
                showlegend=show_legend,
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text="Date", row=1, col=col_idx)
        fig.update_yaxes(title_text="Price (USDT)", row=1, col=col_idx)

    fig.update_layout(
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
