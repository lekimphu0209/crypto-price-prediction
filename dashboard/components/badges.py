"""
Reusable UI badges.

Data provenance matters for trust: anything rendered from mock/demo data
must carry a visible DEMO badge so judges/users never confuse it with real
model output.
"""
import streamlit as st


def demo_badge(text: str = "Dữ liệu mô phỏng (demo) - không phải dữ liệu thật"):
    """Render a prominent badge marking the section as mock/demo data."""
    st.warning(f"🧪 **DEMO** · {text}")


def real_badge(text: str = "Dữ liệu thật từ API"):
    """Render a subtle badge marking real data."""
    st.caption(f"✅ {text}")
