"""
Overview Section Module
Displays real-time prices and historical charts
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
import numpy as np

from .data_providers import get_real_prices, get_historical_data, get_data_providers


def render_overview_section(symbol, timeframe):
    """Render Overview section"""
    st.title("🏠 Market Overview")
    st.markdown("---")
    
    binance, _ = get_data_providers()
    prices = get_real_prices(binance)
    
    # Price metrics
    col1, col2 = st.columns(2)
    with col1:
        if prices and prices['btc_price']:
            st.metric("Bitcoin (BTC)", f"${prices['btc_price']:,.2f}", 
                     f"{prices['btc_change']:+.2f}%", 
                     delta_color="normal" if prices['btc_change'] >= 0 else "inverse")
        else:
            st.metric("Bitcoin (BTC)", "Loading...", "")
    
    with col2:
        if prices and prices['eth_price']:
            st.metric("Ethereum (ETH)", f"${prices['eth_price']:,.2f}", 
                     f"{prices['eth_change']:+.2f}%",
                     delta_color="normal" if prices['eth_change'] >= 0 else "inverse")
        else:
            st.metric("Ethereum (ETH)", "Loading...", "")
    
    # Historical price chart
    st.subheader(f"{symbol} Price Chart")
    
    hist_df = get_historical_data(binance, f"{symbol}USDT", days=30)
    
    if hist_df is not None and len(hist_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df['open'],
            high=hist_df['high'],
            low=hist_df['low'],
            close=hist_df['close'],
            name=symbol
        ))
        fig.update_layout(
            title=f"{symbol} Price - {timeframe}",
            height=400,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to fetch historical data")
