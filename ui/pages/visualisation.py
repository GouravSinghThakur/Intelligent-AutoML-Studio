"""
ui.pages.visualisation – Visualisation page.
"""

from __future__ import annotations

import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src import config
from src.models import compute_metrics, compute_regression_metrics, evaluate_all_models
from src.visualisations import (
    actual_vs_predicted_chart,
    confusion_matrices_chart,
    model_comparison_chart,
    regression_comparison_chart,
    residual_plot,
    roc_curves_chart,
)
from ui.helpers import classification_report_df, require_data, require_model


def render() -> None:
    """Render the Visualisation page."""
    st.title("📊 Visualisation")

    if not require_data() or not require_model():
        st.stop()

    processed  = st.session_state.processed_data
    features   = st.session_state.features
    target     = st.session_state.target
    task_type  = st.session_state.task_type

    X = processed[features]
    y = processed[target]
    stratify_arg = y if (task_type == config.TASK_CLASSIFICATION and y.nunique() <= 20) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_arg,
    )

    fitted = st.session_state.fitted_models

    if not fitted:
        st.info("ℹ️ No pre-trained models found. Running quick training…")
        with st.spinner("Training all models for comparison…"):
            fitted, _ = evaluate_all_models(X_train, y_train, X_test, y_test, task_type=task_type)
            st.session_state.fitted_models = fitted

    # ── Build metrics table ──
    records = []
    if task_type == config.TASK_CLASSIFICATION:
        is_bin = y.nunique() == 2
        for name, pipeline in fitted.items():
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") and is_bin else None
            m = compute_metrics(y_test.values, y_pred, y_prob, is_bin)
            records.append({"Model": name, **m})
    else:
        for name, pipeline in fitted.items():
            y_pred = pipeline.predict(X_test)
            m = compute_regression_metrics(y_test.values, y_pred)
            records.append({"Model": name, **m})

    import pandas as pd
    metrics_df = pd.DataFrame(records)

    # ── Classification tabs ──
    if task_type == config.TASK_CLASSIFICATION:
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Model Comparison", "ROC Curves", "Confusion Matrices", "Best Model Detail"
        ])

        with viz_tab1:
            st.subheader("Model Performance Comparison")
            st.dataframe(metrics_df.round(4), use_container_width=True)
            st.plotly_chart(model_comparison_chart(metrics_df), use_container_width=True)

        with viz_tab2:
            fig_roc = roc_curves_chart(fitted, X_test, y_test)
            if fig_roc:
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curves are only available for binary classification tasks.")

        with viz_tab3:
            st.pyplot(confusion_matrices_chart(fitted, X_test, y_test), use_container_width=True)

        with viz_tab4:
            st.subheader(f"Best Model: `{st.session_state.model_name}`")
            best = st.session_state.model
            y_pred_best = best.predict(X_test)
            is_binary   = y.nunique() == 2

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred_best):.2%}")
            c2.metric("Precision", f"{precision_score(y_test, y_pred_best, average='binary' if is_binary else 'weighted', zero_division=0):.2%}")
            c3.metric("Recall",    f"{recall_score(y_test, y_pred_best, average='binary' if is_binary else 'weighted', zero_division=0):.2%}")
            c4.metric("F1-Score",  f"{f1_score(y_test, y_pred_best, average='binary' if is_binary else 'weighted', zero_division=0):.2%}")

            if is_binary and hasattr(best, "predict_proba"):
                y_prob_best = best.predict_proba(X_test)[:, 1]
                st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob_best):.4f}")

            st.subheader("Classification Report")
            st.dataframe(classification_report_df(y_test, y_pred_best), use_container_width=True)

    # ── Regression tabs ──
    else:
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Model Comparison", "Actual vs Predicted", "Residual Plots", "Predictions Overlay"
        ])

        with viz_tab1:
            st.subheader("Model Performance Comparison")
            st.dataframe(metrics_df.round(4), use_container_width=True)
            st.plotly_chart(model_comparison_chart(metrics_df), use_container_width=True)

        with viz_tab2:
            best = st.session_state.model
            y_pred_best = best.predict(X_test)
            st.plotly_chart(
                actual_vs_predicted_chart(y_test.values, y_pred_best, st.session_state.model_name),
                use_container_width=True,
            )

        with viz_tab3:
            best = st.session_state.model
            y_pred_best = best.predict(X_test)
            st.plotly_chart(
                residual_plot(y_test.values, y_pred_best, st.session_state.model_name),
                use_container_width=True,
            )

        with viz_tab4:
            st.plotly_chart(
                regression_comparison_chart(fitted, X_test, y_test),
                use_container_width=True,
            )
