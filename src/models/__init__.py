"""src.models – Model registry, training, and metrics."""

from src.models.registry import (
    detect_task_type,
    get_classification_configs,
    get_model_configs,
    get_regression_configs,
)
from src.models.metrics import compute_metrics, compute_regression_metrics
from src.models.training import auto_train, evaluate_all_models, train_single_model

__all__ = [
    "auto_train",
    "compute_metrics",
    "compute_regression_metrics",
    "detect_task_type",
    "evaluate_all_models",
    "get_classification_configs",
    "get_model_configs",
    "get_regression_configs",
    "train_single_model",
]
