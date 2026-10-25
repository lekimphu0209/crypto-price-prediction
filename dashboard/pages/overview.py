"""
Overview page - real-time prices + historical charts for BTC & ETH.

Render contract: expose render() with no arguments; timeframe selector lives
inside this page so global state never leaks into other pages.
"""
import streamlit as st

from config import TIMEFRAME_DAYS
from pages._overview_sections import render_overview_section


def render():
    timeframe = st.selectbox("Timeframe", list(TIMEFRAME_DAYS), index=1)
    render_overview_section(timeframe)
