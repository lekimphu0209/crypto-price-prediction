"""
Model selector component.

The dropdown is populated from GET /api/models (the trained-model registry),
so newly trained models appear automatically - no dashboard code change.
"""
import streamlit as st

from services.api_client import available_model_names


def model_selector(key: str = "model_selector") -> str:
    """Render a model dropdown fed by the API; returns the selected model name."""
    names = available_model_names()
    return st.selectbox("Model", names, index=0, key=key)
