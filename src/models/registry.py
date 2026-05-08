"""
src.models.registry – Model configs, Optuna search spaces, and task-type detection.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src import config


def detect_task_type(y: pd.Series, max_unique_ratio: float = 0.05) -> str:
    """Auto-detect whether the target column represents classification or regression.

    Heuristics:
    - If dtype is object/category/bool → classification.
    - If number of unique values ≤ 20 → classification (covers binary, multi-class).
    - If unique ratio ≤ 5% of total rows → classification (high-cardinality integer codes).
    - Otherwise → regression.
    """
    if y.dtype in ("object", "category", "bool"):
        return config.TASK_CLASSIFICATION
    n_unique = y.nunique()
    if n_unique <= 20:
        return config.TASK_CLASSIFICATION
    ratio = n_unique / len(y) if len(y) > 0 else 0
    if ratio <= max_unique_ratio:
        return config.TASK_CLASSIFICATION
    return config.TASK_REGRESSION


def get_classification_configs() -> Dict[str, Dict[str, Any]]:
    """Return fresh classification model configs."""
    return {
        "Logistic Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=config.RANDOM_STATE)),
            ]),
            "grid_params": {
                "classifier__penalty": ["l1", "l2"],
                "classifier__C": [0.01, 0.1, 1.0],
                "classifier__max_iter": [200, 500],
                "classifier__solver": ["saga"],
            },
        },
        "Support Vector Machine": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, random_state=config.RANDOM_STATE)),
            ]),
            "grid_params": {
                "classifier__C": [0.01, 0.1, 1.0],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto"],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
            ]),
            "grid_params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
            },
        },
        "XGBoost": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", XGBClassifier(
                    random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS,
                    eval_metric="logloss", verbosity=0,
                    device=config.XGBOOST_DEVICE, tree_method="hist")),
            ]),
            "grid_params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5, 7],
                "classifier__subsample": [0.8, 1.0],
            },
        },
        "K-Nearest Neighbours": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_jobs=config.N_JOBS)),
            ]),
            "grid_params": {
                "classifier__n_neighbors": [3, 5, 7, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        },
        "Gradient Boosting": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GradientBoostingClassifier(random_state=config.RANDOM_STATE)),
            ]),
            "grid_params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__subsample": [0.8, 1.0],
            },
        },
        "Extra Trees": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", ExtraTreesClassifier(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
            ]),
            "grid_params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        },
    }


def get_regression_configs() -> Dict[str, Dict[str, Any]]:
    """Return fresh regression model configs."""
    return {
        "Ridge Regression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(random_state=config.RANDOM_STATE)),
            ]),
            "grid_params": {"regressor__alpha": [0.01, 0.1, 1.0, 10.0]},
        },
        "SVR": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", SVR()),
            ]),
            "grid_params": {
                "regressor__C": [0.1, 1.0, 10.0],
                "regressor__kernel": ["linear", "rbf"],
                "regressor__gamma": ["scale", "auto"],
            },
        },
        "Random Forest Regressor": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
            ]),
            "grid_params": {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [None, 10, 20],
                "regressor__min_samples_split": [2, 5],
            },
        },
        "XGBoost Regressor": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", XGBRegressor(
                    random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS,
                    verbosity=0, device=config.XGBOOST_DEVICE, tree_method="hist")),
            ]),
            "grid_params": {
                "regressor__n_estimators": [100, 200],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [3, 5, 7],
                "regressor__subsample": [0.8, 1.0],
            },
        },
        "KNN Regressor": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", KNeighborsRegressor(n_jobs=config.N_JOBS)),
            ]),
            "grid_params": {
                "regressor__n_neighbors": [3, 5, 7, 11],
                "regressor__weights": ["uniform", "distance"],
                "regressor__metric": ["euclidean", "manhattan"],
            },
        },
        "Gradient Boosting Regressor": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", GradientBoostingRegressor(random_state=config.RANDOM_STATE)),
            ]),
            "grid_params": {
                "regressor__n_estimators": [100, 200],
                "regressor__learning_rate": [0.05, 0.1],
                "regressor__max_depth": [3, 5],
                "regressor__subsample": [0.8, 1.0],
            },
        },
        "Extra Trees Regressor": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", ExtraTreesRegressor(random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)),
            ]),
            "grid_params": {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [None, 10, 20],
                "regressor__min_samples_split": [2, 5],
            },
        },
    }


def get_model_configs(task_type: str = config.TASK_CLASSIFICATION) -> Dict[str, Dict[str, Any]]:
    """Return the appropriate model registry for the given task type."""
    if task_type == config.TASK_REGRESSION:
        return get_regression_configs()
    return get_classification_configs()
