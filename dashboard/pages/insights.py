"""
Insights page (DEMO) - sentiment, news, external factors, feature importance.

Everything on this page is simulated data kept for UI/UX demonstration.
Every tab carries a DEMO badge so the provenance is unambiguous.
"""
import streamlit as st

from components.badges import demo_badge
from pages._insights_sections import (
    render_sentiment_section,
    render_news_section,
    render_external_factors_section,
    render_feature_importance_section,
    render_backtesting_section,
)


def render():
    st.title("🧪 Insights")
    demo_badge("Toàn bộ nội dung trang này là dữ liệu mô phỏng phục vụ demo giao diện.")

    tab_sentiment, tab_news, tab_external, tab_importance, tab_backtest = st.tabs(
        ["😊 Sentiment", "📰 News", "🌎 External Factors", "📊 Feature Importance", "💰 Backtesting"]
    )
    with tab_sentiment:
        render_sentiment_section()
    with tab_news:
        render_news_section()
    with tab_external:
        render_external_factors_section()
    with tab_importance:
        render_feature_importance_section()
    with tab_backtest:
        render_backtesting_section()
