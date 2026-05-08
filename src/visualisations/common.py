"""
src.visualisations.common – Shared chart builders (used by both classification and regression).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src import config

_PLOTLY_TEMPLATE = "plotly_dark"
_FONT_FAMILY     = "Inter, sans-serif"


def _base_layout(**kwargs) -> dict:
    return dict(
        template=_PLOTLY_TEMPLATE,
        font=dict(family=_FONT_FAMILY, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        **kwargs,
    )


def correlation_heatmap(data: pd.DataFrame) -> plt.Figure:
    """Return a Matplotlib figure with a styled correlation heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, data.shape[1]), max(6, data.shape[1] - 1)))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    mask = np.triu(np.ones_like(data.corr(), dtype=bool))
    sns.heatmap(
        data.corr(), mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, annot_kws={"size": 9, "color": "white"},
        linewidths=0.5, linecolor="#1E2130",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2130")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color="white")
    plt.setp(ax.get_yticklabels(), color="white")
    ax.set_title("Feature Correlation Matrix", color="white", fontsize=14, pad=12)
    plt.tight_layout()
    return fig


def feature_distributions(data: pd.DataFrame, target_col: Optional[str] = None) -> go.Figure:
    """Interactive histograms for all numeric columns."""
    num_cols = data.select_dtypes(include="number").columns.tolist()
    if target_col and target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]

    n_cols = 3
    n_rows = max(1, (len(num_cols) + n_cols - 1) // n_cols)

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=num_cols)

    for i, col in enumerate(num_cols):
        row, col_idx = divmod(i, n_cols)
        trace = go.Histogram(
            x=data[col], name=col,
            marker_color=config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)],
            opacity=0.85,
        )
        fig.add_trace(trace, row=row + 1, col=col_idx + 1)

    fig.update_layout(
        **_base_layout(title_text="Feature Distributions", showlegend=False),
        height=n_rows * 250,
    )
    return fig


def model_comparison_chart(results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing all metrics across models."""
    df = results_df.reset_index() if "Model" not in results_df.columns else results_df.copy()
    if "Model" not in df.columns and df.index.name == "Model":
        df = df.reset_index()

    metric_cols = [c for c in df.columns if c not in ("Model", "CV Score", "index")]
    melted = df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")

    fig = px.bar(
        melted.dropna(subset=["Score"]),
        x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=config.COLOR_PALETTE,
        title="📊 Model Performance Comparison",
        labels={"Score": "Score", "Model": ""},
    )
    fig.update_layout(
        **_base_layout(legend_title_text="Metric"),
        xaxis=dict(gridcolor="#2A2E3F"),
    )
    return fig


def feature_importance_chart(feature_names: List[str], importances: np.ndarray) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#6C63FF", "#FF6584"],
        title="🌟 Feature Importances",
    )
    fig.update_layout(
        **_base_layout(coloraxis_showscale=False),
        xaxis=dict(gridcolor="#2A2E3F"),
        yaxis=dict(gridcolor="#2A2E3F"),
    )
    return fig


def target_distribution_chart(y: pd.Series, label_encoders: dict, target_col: str) -> go.Figure:
    """Pie chart of target class distribution."""
    counts = y.value_counts().reset_index()
    counts.columns = ["Class", "Count"]

    if target_col in label_encoders:
        le = label_encoders[target_col]
        counts["Class"] = le.inverse_transform(counts["Class"].astype(int))

    fig = px.pie(
        counts, names="Class", values="Count",
        color_discrete_sequence=config.COLOR_PALETTE,
        title="🎯 Target Class Distribution", hole=0.4,
    )
    fig.update_layout(**_base_layout())
    return fig


def target_histogram_chart(y: pd.Series, target_col: str) -> go.Figure:
    """Histogram of a continuous target variable."""
    fig = px.histogram(
        x=y, nbins=40, color_discrete_sequence=["#6C63FF"],
        title=f"🎯 Target Distribution — {target_col}",
        labels={"x": target_col, "y": "Count"},
    )
    fig.update_layout(
        **_base_layout(showlegend=False),
        xaxis=dict(gridcolor="#2A2E3F"),
        yaxis=dict(gridcolor="#2A2E3F"),
    )
    return fig
