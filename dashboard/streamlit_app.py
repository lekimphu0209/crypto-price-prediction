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

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
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

# Settings
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["BTC", "ETH"])
timeframe = st.sidebar.selectbox("Timeframe", ["7 Days", "30 Days", "90 Days"])
model = st.sidebar.selectbox("Default Model", ["Ensemble", "Linear Regression", "RNN", "LSTM", "BiLSTM", "Transformer"])

# Render selected section
if page == "🏠 Overview":
    render_overview_section(symbol, timeframe)
elif page == "📈 Price Prediction":
    render_prediction_section(symbol)
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
