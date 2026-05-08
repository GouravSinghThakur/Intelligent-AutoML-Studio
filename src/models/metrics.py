"""
src.models.metrics – Classification and regression metric computation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    is_binary: bool,
) -> Dict[str, Optional[float]]:
    """Compute a standard set of classification metrics."""
    avg = "binary" if is_binary else "weighted"
    metrics: Dict[str, Optional[float]] = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1-Score":  f1_score(y_true, y_pred, average=avg, zero_division=0),
        "ROC-AUC":   None,
    }
    if is_binary and y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute a standard set of regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R²":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MSE":  mse,
        "RMSE": float(np.sqrt(mse)),
    }
