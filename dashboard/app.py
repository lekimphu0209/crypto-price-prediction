"""
Crypto Price Prediction Platform - Dashboard entry point.

Modular layout:
    pages/       one file per page, each exposing render()
    components/  reusable widgets (badges, model selector)
    services/    API client + data providers (only place touching the backend)
    config.py    page registry + shared constants

Run:  streamlit run dashboard/app.py
"""
import importlib
import os
import sys

import streamlit as st

# Make `pages`, `components`, `services`, `config` importable no matter the cwd.
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

from config import PAGES  # noqa: E402

st.set_page_config(
    page_title="Crypto Price Prediction Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep metric cards readable in both light and dark mode.
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

st.sidebar.title("📊 Navigation")
page_label = st.sidebar.radio("Select Section", list(PAGES))

module = importlib.import_module(PAGES[page_label])
module.render()

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Crypto Price Prediction Platform | Clean Architecture | SOLID Principles</p>
        <p>Real-time data from Binance · Trend-direction models served by FastAPI</p>
    </div>
""", unsafe_allow_html=True)
