"""
src.config – Centralised application settings.
"""

from typing import List

APP_TITLE = "AutoML Studio"
APP_ICON  = "🤖"
LAYOUT    = "wide"

TEST_SIZE      = 0.20
RANDOM_STATE   = 42
CV_FOLDS       = 5
OPTUNA_TRIALS  = 10
N_JOBS         = -1
SCORING_METRIC_CLF = "accuracy"
SCORING_METRIC_REG = "r2"

# Legacy alias
SCORING_METRIC = SCORING_METRIC_CLF


def _detect_gpu() -> bool:
    try:
        import xgboost as xgb
        import numpy as _np
        bst = xgb.XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
        bst.fit(_np.array([[1, 2]]), _np.array([0]))
        return True
    except Exception:
        return False


USE_GPU        = _detect_gpu()
XGBOOST_DEVICE = "cuda" if USE_GPU else "cpu"

SUPPORTED_FILE_TYPES: List[str] = ["csv", "xlsx", "xls"]

MODEL_EXPORT_FILENAME = "automl_model.joblib"

COLOR_PALETTE = [
    "#6C63FF", "#FF6584", "#43D8C9", "#FFB347",
    "#A8E6CF", "#FF8B94", "#84B1ED", "#FFA07A",
    "#B39DDB", "#4DD0E1", "#E6EE9C", "#F48FB1",
    "#80DEEA", "#CE93D8",
]

BINARY_METRICS     = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
MULTICLASS_METRICS = ["Accuracy", "Precision (W)", "Recall (W)", "F1-Score (W)"]
REGRESSION_METRICS = ["R²", "MAE", "MSE", "RMSE"]

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION     = "regression"
