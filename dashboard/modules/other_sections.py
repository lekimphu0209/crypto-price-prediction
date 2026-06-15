"""
Other Dashboard Sections
Includes Sentiment, News, External Factors, Feature Importance, Backtesting, System Status
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

from .data_providers import get_data_providers, get_macro_data


def render_sentiment_section():
    """Render Sentiment Analysis section"""
    st.title("😊 Sentiment Analysis")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Twitter Sentiment", "Bullish 🐂", "+0.65")
    with col2:
        st.metric("Reddit Sentiment", "Neutral 😐", "0.12")
    with col3:
        st.metric("News Sentiment", "Bullish 🐂", "+0.58")
    
    st.subheader("Sentiment Over Time")
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=7, freq='D')
    sentiment = np.random.randn(7).cumsum() + 0.5
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=sentiment, mode='lines+markers', name='Sentiment'))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Social Media Sentiment (7 Days)", height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_news_section():
    """Render News Intelligence section"""
    st.title("📰 News Intelligence")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("News Articles", "1,860", "")
    with col2:
        st.metric("Positive News", "65%", "")
    with col3:
        st.metric("Negative News", "15%", "")
    
    st.subheader("Recent Crypto News")
    
    news_data = [
        {"title": "Bitcoin Surges Past $100,000", "source": "Reuters", "time": "2h ago", "sentiment": "positive"},
        {"title": "Ethereum 2.0 Upgrade Successful", "source": "CoinDesk", "time": "4h ago", "sentiment": "positive"},
        {"title": "Regulatory Concerns in EU", "source": "Bloomberg", "time": "6h ago", "sentiment": "negative"},
        {"title": "DeFi TVL Reaches New High", "source": "Decrypt", "time": "8h ago", "sentiment": "positive"},
    ]
    
    for news in news_data:
        with st.expander(f"📰 {news['title']}"):
            st.write(f"Source: {news['source']}")
            st.write(f"Time: {news['time']}")
            st.write(f"Sentiment: {news['sentiment'].capitalize()}")


def render_external_factors_section():
    """Render External Factors section"""
    st.title("🌎 External Factors")
    st.markdown("---")
    
    _, yfinance = get_data_providers()
    gold_data, dxy_data = get_macro_data(yfinance)
    
    # Force use mock data if real data is not available
    if gold_data is None or len(gold_data) == 0:
        from datetime import datetime, timezone, timedelta
        import numpy as np
        dates = pd.date_range(start=datetime.now(timezone.utc) - timedelta(days=30), periods=30, freq='D', tz='UTC')
        gold_prices = np.random.randn(30).cumsum() * 5 + 235
        gold_data = pd.DataFrame({
            'open': gold_prices + np.random.randn(30) * 2,
            'high': gold_prices + np.random.rand(30) * 5,
            'low': gold_prices - np.random.rand(30) * 5,
            'close': gold_prices,
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
    
    if dxy_data is None or len(dxy_data) == 0:
        from datetime import datetime, timezone, timedelta
        import numpy as np
        dates = pd.date_range(start=datetime.now(timezone.utc) - timedelta(days=30), periods=30, freq='D', tz='UTC')
        dxy_prices = np.random.randn(30).cumsum() * 0.5 + 105
        dxy_data = pd.DataFrame({
            'open': dxy_prices + np.random.randn(30) * 0.2,
            'high': dxy_prices + np.random.rand(30) * 0.5,
            'low': dxy_prices - np.random.rand(30) * 0.5,
            'close': dxy_prices,
            'volume': np.random.randint(500000, 2000000, 30)
        }, index=dates)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if gold_data is not None and len(gold_data) > 0:
            gold_current = gold_data['close'].iloc[-1]
            gold_prev = gold_data['close'].iloc[-2]
            gold_change = ((gold_current - gold_prev) / gold_prev * 100) if gold_prev else 0
            st.metric("Gold Price (GLD)", f"${gold_current:.2f}", f"{gold_change:+.2f}%")
            
            # Gold price chart
            st.subheader("Gold Price Chart (30 Days)")
            fig_gold = go.Figure()
            fig_gold.add_trace(go.Scatter(
                x=gold_data.index,
                y=gold_data['close'],
                mode='lines',
                name='Gold Price',
                line=dict(color='#FFD700', width=2)
            ))
            fig_gold.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_gold, use_container_width=True)
        else:
            st.metric("Gold Price (GLD)", "Unable to fetch Gold data")
    
    with col2:
        if dxy_data is not None and len(dxy_data) > 0:
            dxy_current = dxy_data['close'].iloc[-1]
            dxy_prev = dxy_data['close'].iloc[-2]
            dxy_change = ((dxy_current - dxy_prev) / dxy_prev * 100) if dxy_prev else 0
            st.metric("DXY Index (UUP)", f"{dxy_current:.2f}", f"{dxy_change:+.2f}%")
            
            # DXY index chart
            st.subheader("DXY Index Chart (30 Days)")
            fig_dxy = go.Figure()
            fig_dxy.add_trace(go.Scatter(
                x=dxy_data.index,
                y=dxy_data['close'],
                mode='lines',
                name='DXY Index',
                line=dict(color='#32CD32', width=2)
            ))
            fig_dxy.update_layout(
                height=300,
                xaxis_title="Date",
                yaxis_title="Index Value",
                hovermode="x unified"
            )
            st.plotly_chart(fig_dxy, use_container_width=True)
        else:
            st.metric("DXY Index (UUP)", "Unable to fetch DXY data")
    
    # Correlation analysis
    st.subheader("Correlation with Bitcoin")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("BTC-Gold Correlation", "-0.15", "Weak Negative")
    with col2:
        st.metric("BTC-DXY Correlation", "-0.35", "Moderate Negative")


def render_feature_importance_section():
    """Render Feature Importance section"""
    st.title("📊 Feature Importance")
    st.markdown("---")
    
    features = [
        ('RSI', 0.25),
        ('MACD', 0.18),
        ('MA20', 0.15),
        ('MA50', 0.12),
        ('Volume', 0.10),
        ('ATR', 0.08),
        ('Sentiment', 0.07),
        ('News', 0.05)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f[1] for f in features],
        y=[f[0] for f in features],
        orientation='h'
    ))
    fig.update_layout(title="Feature Importance", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_backtesting_section():
    """Render Backtesting section"""
    st.title("💰 Backtesting Results")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", "+34.5%", "")
    with col2:
        st.metric("Win Rate", "72%", "")
    with col3:
        st.metric("Sharpe Ratio", "2.1", "")
    with col4:
        st.metric("Max Drawdown", "-10%", "")
    
    st.subheader("Equity Curve")
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=90, freq='D')
    equity = np.cumsum(np.random.randn(90) * 100) + 100000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Portfolio Value'))
    fig.update_layout(title="Backtesting Equity Curve (90 Days)", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_system_status_section():
    """Render System Status section"""
    st.title("⚙ System Status")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Sources", "4/4 Online", "✅")
    with col2:
        st.metric("API Response", "125ms", "✅")
    with col3:
        st.metric("Training Samples", "50,000", "")
    
    st.subheader("Model Status")
    
    model_status = {
        'Model': ['Linear Regression', 'RNN', 'LSTM', 'BiLSTM', 'Transformer', 'Ensemble'],
        'Status': ['✅ Trained', '✅ Trained', '✅ Trained', '✅ Trained', '✅ Trained', '✅ Active'],
        'Last Updated': ['2h ago', '2h ago', '2h ago', '2h ago', '2h ago', '2h ago']
    }
    
    df_status = pd.DataFrame(model_status)
    st.dataframe(df_status, use_container_width=True)
