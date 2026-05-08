"""
src.visualisations.classification – Classification-specific charts.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline

from src import config
from src.visualisations.common import _base_layout


def roc_curves_chart(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Optional[go.Figure]:
    """Return a Plotly ROC-curve figure (binary classification only)."""
    if len(np.unique(y_test)) != 2:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Random Classifier", showlegend=True,
    ))

    for i, (name, pipeline) in enumerate(fitted_models.items()):
        if not hasattr(pipeline, "predict_proba"):
            continue
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC = {roc_auc:.3f})",
                line=dict(color=config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)], width=2),
            ))
        except Exception:
            pass

    fig.update_layout(
        **_base_layout(title="🎯 ROC Curves"),
        xaxis=dict(title="False Positive Rate", range=[0, 1], gridcolor="#2A2E3F"),
        yaxis=dict(title="True Positive Rate", range=[0, 1.05], gridcolor="#2A2E3F"),
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def confusion_matrices_chart(
    fitted_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> plt.Figure:
    """Matplotlib figure with one confusion-matrix heatmap per model."""
    n = len(fitted_models)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor("#0E1117")

    axes_flat = np.array(axes).ravel() if n > 1 else [axes]

    for ax, (name, pipeline) in zip(axes_flat, fitted_models.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, linecolor="#1E2130", annot_kws={"color": "white"},
        )
        ax.set_facecolor("#0E1117")
        ax.set_title(name, color="white", fontsize=13)
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        ax.tick_params(colors="white")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig
