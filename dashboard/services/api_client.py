"""
API client - single place that talks to the FastAPI backend.

Pages/components never call requests directly; they go through these
functions so the base URL, timeouts, and error shape live in one spot.
"""
import requests
import streamlit as st

from config import API_BASE_URL

# Fallback list if the API is offline (mirrors src/models/registry.py)
FALLBACK_MODELS = ["Logistic Regression", "RNN", "LSTM", "BiLSTM", "Transformer"]


@st.cache_data(ttl=300, show_spinner=False)
def predict(symbol: str, model_name: str, display_days: int, horizon: int = None) -> dict:
    """
    Call POST /api/predict/ for a trend-direction prediction.

    Returns the API response dict (status=success/error).
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/predict/",
            json={
                "symbol": symbol,
                "model_name": model_name,
                "display_days": display_days,
                "horizon": horizon,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Không kết nối được backend API. Kiểm tra server đã chạy chưa."}
    except Exception as e:
        return {"status": "error", "message": f"Lỗi API: {e}"}


@st.cache_data(ttl=60, show_spinner=False)
def list_models() -> list:
    """
    Call GET /api/models/ -> list of {name, symbol, type, path, metrics}.

    Returns [] when the API is unreachable.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/api/models/", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return data.get("models", [])
    except Exception:
        pass
    return []


def available_model_names() -> list:
    """Distinct trained model names from the API, falling back to the built-in list."""
    models = list_models()
    names = sorted({m.get("name") for m in models if m.get("name")})
    return names or FALLBACK_MODELS
