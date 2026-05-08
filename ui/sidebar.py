"""
ui.sidebar – Sidebar navigation and status indicators.
"""

from __future__ import annotations

import streamlit as st

from src import config


def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:16px 0'>"
            f"<span style='font-size:2.5rem'>🤖</span>"
            f"<h2 style='margin:8px 0 4px;color:#EAEAF5'>{config.APP_TITLE}</h2>"
            f"<span style='color:#6C63FF;font-size:.8rem;font-weight:600'>POWERED BY OPTUNA</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigation",
            ["🏠 Home", "📤 Data Upload", "🎯 Model Training", "📊 Visualisation", "🔮 Prediction"],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("AutoML Studio v2.0")

        if st.session_state.raw_data is not None:
            st.success(f"✅ Dataset loaded  ({st.session_state.raw_data.shape[0]:,} rows)")
        if st.session_state.model is not None:
            st.success(f"✅ Model trained: {st.session_state.model_name}")
        if config.USE_GPU:
            st.success("🟢 GPU: CUDA active (XGBoost)")
        else:
            st.info("⚪ GPU: Not available (CPU mode)")

    return page
