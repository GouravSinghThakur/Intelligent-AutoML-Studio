"""
ui.helpers – Shared helper functions used across pages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report


def require_data() -> bool:
    """Return True if processed data exists; show warning otherwise."""
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process your dataset first (Data Upload page).")
        return False
    return True


def require_model() -> bool:
    """Return True if a trained model exists; show warning otherwise."""
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first (Model Training page).")
        return False
    return True


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """Return a styled classification report as a DataFrame."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose().round(3)
