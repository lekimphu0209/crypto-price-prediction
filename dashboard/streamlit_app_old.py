"""
Streamlit Dashboard for Crypto Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Crypto Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 2.5rem;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Select Symbol", ["BTCUSDT", "ETHUSDT"])
model = st.sidebar.selectbox("Select Model", ["Ensemble", "Linear Regression", "XGBoost", "LSTM", "BiLSTM", "Transformer"])

# Main content
st.title("🚀 Crypto Price Prediction Dashboard")
st.markdown("---")

# Price Chart
col1, col2, col3 = st.columns(3)

with col1:
    current_price = st.metric("Current Price", "$67,234.50", "+2.3%")

with col2:
    prediction = st.metric("Predicted Price", "$68,500.00", "+1.9%")

with col3:
    confidence = st.metric("Confidence", "85%", "")

# Chart
st.subheader("Price Chart & Predictions")

# Generate sample data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
actual_prices = np.random.randn(100).cumsum() + 65000
predicted_prices = actual_prices + np.random.randn(100) * 500

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=actual_prices, mode='lines', name='Actual Price', line=dict(color='#1f77b4')))
fig.add_trace(go.Scatter(x=dates, y=predicted_prices, mode='lines', name='Predicted Price', line=dict(color='#ff7f0e', dash='dash')))

fig.update_layout(
    title=f"{symbol} Price Chart",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Model Comparison
st.subheader("Model Comparison")

models_data = {
    'Model': ['Linear Regression', 'XGBoost', 'LSTM', 'BiLSTM', 'Transformer', 'Ensemble'],
    'MAE': [234.5, 198.3, 189.2, 185.6, 178.9, 175.2],
    'RMSE': [312.8, 267.4, 245.6, 238.9, 231.2, 225.6],
    'R²': [0.89, 0.91, 0.92, 0.93, 0.94, 0.95]
}

df_models = pd.DataFrame(models_data)
st.dataframe(df_models, use_container_width=True)

# Backtesting Results
st.subheader("Backtesting Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Profit", "+$2,345.67", "+23.5%")

with col2:
    st.metric("Win Rate", "65%", "")

with col3:
    st.metric("Sharpe Ratio", "1.85", "")

with col4:
    st.metric("Max Drawdown", "-8.5%", "")

# Sentiment Analysis
st.subheader("Sentiment Analysis")

sentiment_data = {
    'Source': ['Twitter/X', 'Reddit', 'News', 'Overall'],
    'Sentiment Score': [0.65, 0.58, 0.72, 0.65],
    'Volume': [12500, 8900, 4500, 25900]
}

df_sentiment = pd.DataFrame(sentiment_data)

col1, col2 = st.columns(2)

with col1:
    fig_sentiment = go.Figure(data=[
        go.Bar(name='Sentiment Score', x=df_sentiment['Source'], y=df_sentiment['Sentiment Score'])
    ])
    fig_sentiment.update_layout(title="Sentiment Scores", yaxis_range=[0, 1])
    st.plotly_chart(fig_sentiment, use_container_width=True)

with col2:
    fig_volume = go.Figure(data=[
        go.Bar(name='Volume', x=df_sentiment['Source'], y=df_sentiment['Volume'])
    ])
    fig_volume.update_layout(title="Post Volume")
    st.plotly_chart(fig_volume, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Crypto Price Prediction Platform | Clean Architecture | SOLID Principles</p>
    </div>
""", unsafe_allow_html=True)
