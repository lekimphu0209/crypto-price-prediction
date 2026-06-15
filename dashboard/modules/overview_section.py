"""
Overview Section Module
Displays real-time prices and historical charts
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .data_providers import (
    get_real_prices,
    get_historical_data,
    get_data_providers,
    timeframe_to_days,
)

SYMBOL_COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
}


def render_overview_section(timeframe):
    """Render Market Overview section with both BTC and ETH"""
    st.title("🏠 Market Overview")
    st.markdown("---")
    
    binance, _ = get_data_providers()
    prices = get_real_prices(binance)
    days = timeframe_to_days(timeframe)

    # Price metrics for both BTC and ETH
    col1, col2 = st.columns(2)
    with col1:
        if prices and prices["btc_price"]:
            st.metric(
                "Bitcoin (BTC)",
                f"${prices['btc_price']:,.2f}",
                f"{prices['btc_change']:+.2f}%",
                delta_color="normal" if prices["btc_change"] >= 0 else "inverse",
            )
        else:
            st.metric("Bitcoin (BTC)", "Loading...", "")

    with col2:
        if prices and prices["eth_price"]:
            st.metric(
                "Ethereum (ETH)",
                f"${prices['eth_price']:,.2f}",
                f"{prices['eth_change']:+.2f}%",
                delta_color="normal" if prices["eth_change"] >= 0 else "inverse",
            )
        else:
            st.metric("Ethereum (ETH)", "Loading...", "")

    # BTC and ETH Price Charts side by side (single subplot figure)
    st.subheader(f"Price Charts — {timeframe}")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Bitcoin (BTC)", "Ethereum (ETH)"),
        horizontal_spacing=0.08,
    )

    for col_idx, (symbol, binance_symbol) in enumerate(
        [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")], start=1
    ):
        hist_df = get_historical_data(binance, binance_symbol, days=days)
        if hist_df is not None and len(hist_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=hist_df.index,
                    y=hist_df["close"],
                    mode="lines",
                    name=f"{symbol} Close",
                    showlegend=False,
                    line=dict(color=SYMBOL_COLORS.get(symbol, "#1f77b4"), width=2),
                ),
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(title_text="Date", row=1, col=col_idx)
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=col_idx)

    fig.update_layout(
        height=350,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
