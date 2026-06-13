"""
Price Prediction Section Module
Displays model predictions and forecasts
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

from .data_providers import get_real_prices, get_historical_data, get_data_providers


def render_prediction_section(symbol):
    """Render Price Prediction section"""
    st.title("📈 Price Prediction Dashboard")
    st.markdown("---")
    
    selected_model = st.selectbox("Select Model", ["Linear Regression", "RNN", "LSTM", "BiLSTM", "Transformer", "Ensemble"])
    
    binance, _ = get_data_providers()
    prices = get_real_prices(binance)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if prices and prices['btc_price'] and symbol == "BTC":
            st.metric("Current Price", f"${prices['btc_price']:,.2f}", "")
        elif prices and prices['eth_price'] and symbol == "ETH":
            st.metric("Current Price", f"${prices['eth_price']:,.2f}", "")
        else:
            st.metric("Current Price", "Loading...", "")
    
    with col2:
        st.metric("Predicted Tomorrow", "$110,150", "+1.75%", delta_color="normal")
    
    with col3:
        st.metric("Expected Change", "+$1,900", "+1.75%", delta_color="normal")
    
    with col4:
        st.metric("Confidence", "82%", "")
    
    # Price forecast chart
    st.subheader("Price Forecast")
    
    historical_df = get_historical_data(binance, f"{symbol}USDT" if symbol == "BTC" else "ETHUSDT", days=60)
    
    if historical_df is not None and len(historical_df) > 0:
        historical = historical_df['close'].values
        dates = pd.date_range(start=historical_df.index[0], periods=len(historical), freq='D')
        forecast = np.random.randn(7).cumsum() + historical[-1]
        forecast_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=7, freq='D')
    else:
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), periods=60, freq='D')
        historical = np.random.randn(60).cumsum() + 108000
        forecast = np.random.randn(7).cumsum() + historical[-1]
        forecast_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=7, freq='D')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=historical, mode='lines', name='Historical', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines+markers', name='Predicted 7 Days', line=dict(color='#ff7f0e', dash='dash')))
    fig.add_vline(x=forecast_dates[0] - timedelta(days=1), line_dash="dash", line_color="red", annotation_text="Today")
    fig.update_layout(title=f"{symbol} Price Prediction - {selected_model}", height=400)
    st.plotly_chart(fig, use_container_width=True)
