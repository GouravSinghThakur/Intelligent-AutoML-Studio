"""
app.py – AutoML Studio main Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import warnings

import streamlit as st

from src import config
from ui.styles import inject_styles
from ui.sidebar import render_sidebar
from ui.pages import home, data_upload, model_training, visualisation, prediction

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={"About": "AutoML Studio – industry-grade AutoML powered by Optuna & scikit-learn"},
)

inject_styles()

_STATE_DEFAULTS = {
    "raw_data":        None,
    "processed_data":  None,
    "label_encoders":  {},
    "model":           None,
    "model_name":      None,
    "features":        [],
    "target":          None,
    "X_test":          None,
    "y_test":          None,
    "auto_results_df": None,
    "fitted_models":   {},
    "task_type":       config.TASK_CLASSIFICATION,
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

page = render_sidebar()

if page == "🏠 Home":
    home.render()
elif page == "📤 Data Upload":
    data_upload.render()
elif page == "🎯 Model Training":
    model_training.render()
elif page == "📊 Visualisation":
    visualisation.render()
elif page == "🔮 Prediction":
    prediction.render()
