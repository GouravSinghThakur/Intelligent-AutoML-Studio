"""
ui.pages.home – Home / landing page.
"""

from __future__ import annotations

import streamlit as st

from src import config


def render() -> None:
    """Render the Home page."""
    st.markdown('<div class="hero-badge">AutoML · Optuna · scikit-learn</div>', unsafe_allow_html=True)
    st.title("🚀 AutoML Studio")
    st.markdown(
        "<p style='font-size:1.15rem;color:#A0AEC0;max-width:700px'>"
        "An industry-grade AutoML platform that automates the entire machine-learning lifecycle — "
        "from raw data upload to hyperparameter-optimised model deployment."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Algorithms", "14")
    with col2:
        st.metric("Task Types", "Clf + Reg")
    with col3:
        st.metric("Tuning Engine", "Optuna")
    with col4:
        st.metric("Max Trials / Model", str(config.OPTUNA_TRIALS))
    with col5:
        st.metric("CV Folds", str(config.CV_FOLDS))

    st.markdown("---")
    st.subheader("✨ Features")

    cards = [
        ("🔍 Smart Preprocessing", "Auto-imputes missing values, encodes categoricals, and drops duplicates — zero config required."),
        ("🧠 Auto Task Detection", "Automatically detects whether your target is classification or regression — no manual setup."),
        ("⚙️ Automated Hyperparameter Tuning", "Optuna's Bayesian optimisation explores the search space intelligently across 14 algorithms."),
        ("📈 Comprehensive Evaluation", "Classification: Accuracy, Precision, Recall, F1, ROC-AUC. Regression: R², MAE, MSE, RMSE."),
        ("🚀 One-Click Export", "Download the best fitted Pipeline (scaler included) as a .joblib file ready for production."),
        ("🎯 Instant Predictions", "Enter feature values manually and get predictions — class probabilities or regression estimates."),
        ("📊 Interactive Visualisations", "Plotly-powered charts with dark-mode theming: ROC curves, residual plots, and more."),
        ("🔀 Dual-Mode Engine", "7 classification models (Logistic Regression, SVM, RF, XGBoost, KNN, GBM, Extra Trees) + 7 regression models."),
    ]

    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(cards):
                title, desc = cards[i + j]
                col.markdown(
                    f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("🚦 Getting Started")
    st.markdown("""
1. **Upload** your CSV / Excel file on the **Data Upload** page.
2. **Select** features and target column, then click **Auto Train** on the **Model Training** page.
3. **Explore** model performance on the **Visualisation** page.
4. **Predict** on new data or **download** the trained model.
""")
