"""
src.visualisations.regression – Regression-specific charts.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

from src import config
from src.visualisations.common import _base_layout


def actual_vs_predicted_chart(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Scatter plot of actual vs predicted values."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color="#6C63FF", size=6, opacity=0.6, line=dict(width=0.5, color="#EAEAF5")),
        name="Predictions",
    ))

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(dash="dash", color="#FF6584", width=2),
        name="Perfect Prediction",
    ))

    fig.update_layout(
        **_base_layout(title=f"🎯 Actual vs Predicted — {model_name}"),
        xaxis=dict(title="Actual Values", gridcolor="#2A2E3F"),
        yaxis=dict(title="Predicted Values", gridcolor="#2A2E3F"),
    )
    return fig


def residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Residual plot for regression models."""
    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color="#43D8C9", size=6, opacity=0.6, line=dict(width=0.5, color="#EAEAF5")),
        name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#FF6584", line_width=2,
                  annotation_text="Zero Error", annotation_position="top right")

    fig.update_layout(
        **_base_layout(title=f"📉 Residual Plot — {model_name}"),
        xaxis=dict(title="Predicted Values", gridcolor="#2A2E3F"),
        yaxis=dict(title="Residuals (Actual − Predicted)", gridcolor="#2A2E3F"),
    )
    return fig


def regression_comparison_chart(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> go.Figure:
    """Overlay actual vs predicted for all regression models."""
    fig = go.Figure()

    sort_idx = np.argsort(y_test.values)
    x_axis = np.arange(len(y_test))

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_test.values[sort_idx],
        mode="lines", line=dict(color="#EAEAF5", width=2), name="Actual",
    ))

    for i, (name, pipeline) in enumerate(fitted_models.items()):
        y_pred = pipeline.predict(X_test)
        fig.add_trace(go.Scatter(
            x=x_axis, y=y_pred[sort_idx], mode="lines",
            line=dict(color=config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)], width=1.5),
            name=name, opacity=0.8,
        ))

    fig.update_layout(
        **_base_layout(title="📈 Predictions Overlay (sorted by actual)"),
        xaxis=dict(title="Sample Index (sorted)", gridcolor="#2A2E3F"),
        yaxis=dict(title="Target Value", gridcolor="#2A2E3F"),
    )
    return fig
