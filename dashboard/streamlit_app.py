"""
Crypto Price Prediction Platform - Main Dashboard
Modular architecture with separated sections
"""
import streamlit as st
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Crypto Price Prediction Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — use Streamlit theme variables so metrics stay readable in light/dark mode
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    div[data-testid="stMetric"] label p,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Import modules
from modules.overview_section import render_overview_section
from modules.prediction_section import render_prediction_section
from modules.comparison_section import render_comparison_section
from modules.other_sections import (
    render_sentiment_section,
    render_news_section,
    render_external_factors_section,
    render_feature_importance_section,
    render_backtesting_section,
    render_system_status_section
)

# Header with settings only
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    timeframe = st.selectbox("Timeframe", ["7 Days", "30 Days", "90 Days"], index=1)

with col2:
    model = st.selectbox(
        "Model",
        ["Linear Regression", "RNN", "LSTM", "BiLSTM", "Transformer"],
        index=0
    )

st.markdown("---")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "📈 Price Prediction",
        "🤖 Model Comparison",
        "😊 Sentiment Analysis",
        "📰 News Intelligence",
        "🌎 External Factors",
        "📊 Feature Importance",
        "💰 Backtesting",
        "⚙ System Status"
    ]
)

# Render selected section
if page == "🏠 Overview":
    render_overview_section(timeframe)
elif page == "📈 Price Prediction":
    render_prediction_section(timeframe, model)
elif page == "🤖 Model Comparison":
    render_comparison_section()
elif page == "😊 Sentiment Analysis":
    render_sentiment_section()
elif page == "📰 News Intelligence":
    render_news_section()
elif page == "🌎 External Factors":
    render_external_factors_section()
elif page == "📊 Feature Importance":
    render_feature_importance_section()
elif page == "💰 Backtesting":
    render_backtesting_section()
elif page == "⚙ System Status":
    render_system_status_section()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Crypto Price Prediction Platform | Clean Architecture | SOLID Principles</p>
        <p>Real-time data from Binance, Yahoo Finance, NewsAPI, and X.com</p>
    </div>
""", unsafe_allow_html=True)
