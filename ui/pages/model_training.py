"""
ui.pages.model_training – Model Training page.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src import config
from src.models import auto_train, detect_task_type
from src.visualisations import (
    actual_vs_predicted_chart,
    feature_importance_chart,
    model_comparison_chart,
    target_distribution_chart,
    target_histogram_chart,
)
from ui.helpers import classification_report_df, require_data


def render() -> None:
    """Render the Model Training page."""
    st.title("🎯 Model Training")

    if not require_data():
        st.stop()

    processed = st.session_state.processed_data
    columns   = processed.columns.tolist()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        features = st.multiselect(
            "Select feature columns",
            columns,
            default=columns[:-1],
            help="All columns that will be used as model inputs.",
        )
    with col_right:
        target = st.selectbox(
            "Select target column",
            columns,
            index=len(columns) - 1,
            help="The column you want to predict.",
        )

    if not features or target in features:
        st.warning("⚠️ Please select at least one feature that is not the target column.")
        st.stop()

    y_series = processed[target]
    task_type = detect_task_type(y_series)

    if task_type == config.TASK_CLASSIFICATION:
        st.success(f"🧠 Auto-detected task: **Classification** ({y_series.nunique()} classes)")
        with st.expander("📋 Target class distribution", expanded=False):
            st.plotly_chart(
                target_distribution_chart(y_series, st.session_state.label_encoders, target),
                use_container_width=True,
            )
    else:
        st.success("🧠 Auto-detected task: **Regression** (continuous target)")
        with st.expander("📋 Target distribution", expanded=False):
            st.plotly_chart(
                target_histogram_chart(y_series, target),
                use_container_width=True,
            )

    st.markdown("---")

    train_col, _ = st.columns([1, 3])
    with train_col:
        run_training = st.button("🚀 Auto Train All Models", type="primary", use_container_width=True)

    if run_training:
        X = processed[features]
        y = processed[target]

        stratify_arg = y if (task_type == config.TASK_CLASSIFICATION and y.nunique() <= 20) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=stratify_arg,
        )

        st.info(f"🔀 Train split: **{len(X_train):,}** samples | Test split: **{len(X_test):,}** samples")

        best_model, best_model_name, results_df, all_pipelines = auto_train(
            X_train, y_train, X_test, y_test,
            n_trials=config.OPTUNA_TRIALS,
            task_type=task_type,
        )

        if best_model is None:
            st.error("❌ All models failed to train. Check your data and try again.")
            st.stop()

        st.session_state.model           = best_model
        st.session_state.model_name      = best_model_name
        st.session_state.features        = features
        st.session_state.target          = target
        st.session_state.X_test          = X_test
        st.session_state.y_test          = y_test
        st.session_state.auto_results_df = results_df
        st.session_state.fitted_models   = all_pipelines
        st.session_state.task_type       = task_type

        st.subheader("📊 AutoTrain Results")
        st.dataframe(results_df, use_container_width=True)
        st.plotly_chart(model_comparison_chart(results_df), use_container_width=True)

        st.markdown(f"### 🏆 Best Model: `{best_model_name}`")
        y_pred = best_model.predict(X_test)

        if task_type == config.TASK_CLASSIFICATION:
            is_binary = y.nunique() == 2
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
            m2.metric("Precision", f"{precision_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0):.2%}")
            m3.metric("Recall",    f"{recall_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0):.2%}")
            m4.metric("F1-Score",  f"{f1_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0):.2%}")

            st.subheader("📋 Classification Report")
            st.dataframe(classification_report_df(y_test, y_pred), use_container_width=True)
        else:
            mse_val = mean_squared_error(y_test, y_pred)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R²",   f"{r2_score(y_test, y_pred):.4f}")
            m2.metric("MAE",  f"{mean_absolute_error(y_test, y_pred):.4f}")
            m3.metric("MSE",  f"{mse_val:.4f}")
            m4.metric("RMSE", f"{np.sqrt(mse_val):.4f}")

            st.plotly_chart(
                actual_vs_predicted_chart(y_test.values, y_pred, best_model_name),
                use_container_width=True,
            )

        # Feature importances for tree-based models
        step_name = "classifier" if task_type == config.TASK_CLASSIFICATION else "regressor"
        estimator = best_model.named_steps.get(step_name)
        if estimator is not None and hasattr(estimator, "feature_importances_"):
            st.subheader("🌟 Feature Importances")
            fig = feature_importance_chart(features, estimator.feature_importances_)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("💾 Export Trained Model")

        model_bundle = {
            "model":          best_model,
            "model_name":     best_model_name,
            "task_type":      task_type,
            "label_encoders": st.session_state.label_encoders,
            "features":       features,
            "target":         target,
            "config": {
                "test_size":     config.TEST_SIZE,
                "random_state":  config.RANDOM_STATE,
                "scoring":       config.SCORING_METRIC_CLF if task_type == config.TASK_CLASSIFICATION else config.SCORING_METRIC_REG,
            },
        }
        buffer = io.BytesIO()
        joblib.dump(model_bundle, buffer)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download model (.joblib)",
            data=buffer,
            file_name=config.MODEL_EXPORT_FILENAME,
            mime="application/octet-stream",
        )
