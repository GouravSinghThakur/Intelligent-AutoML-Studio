"""
src.models.training – GridSearchCV, Optuna tuning, and auto-train orchestration.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import streamlit as st
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from src import config
from src.models.metrics import compute_metrics, compute_regression_metrics
from src.models.registry import get_model_configs

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str = config.TASK_CLASSIFICATION,
) -> Tuple[Pipeline, float]:
    """Train one model with GridSearchCV and return the best estimator + CV score."""
    configs = get_model_configs(task_type)
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = configs[model_name]
    scoring = (config.SCORING_METRIC_CLF
               if task_type == config.TASK_CLASSIFICATION
               else config.SCORING_METRIC_REG)

    if task_type == config.TASK_CLASSIFICATION:
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    else:
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    gs = GridSearchCV(
        estimator=cfg["pipeline"],
        param_grid=cfg["grid_params"],
        cv=cv,
        n_jobs=config.N_JOBS,
        scoring=scoring,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info("%s best CV score: %.4f — params: %s", model_name, gs.best_score_, gs.best_params_)
    return gs.best_estimator_, gs.best_score_


def _build_optuna_params(trial: optuna.Trial, model_name: str, task_type: str) -> Dict[str, Any]:
    """Return a hyperparameter dict for the given model, sampled by Optuna."""
    if task_type == config.TASK_CLASSIFICATION:
        if model_name == "Logistic Regression":
            return {
                "classifier__penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "classifier__C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "classifier__solver": trial.suggest_categorical("solver", ["saga"]),
                "classifier__max_iter": trial.suggest_int("max_iter", 200, 1000),
            }
        if model_name == "Support Vector Machine":
            return {
                "classifier__C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "classifier__kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                "classifier__gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
        if model_name == "Random Forest":
            return {
                "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "classifier__max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                "classifier__min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "classifier__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "classifier__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        if model_name == "XGBoost":
            return {
                "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "classifier__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "classifier__max_depth": trial.suggest_int("max_depth", 2, 10),
                "classifier__min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "classifier__gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            }
        if model_name == "K-Nearest Neighbours":
            return {
                "classifier__n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "classifier__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "classifier__metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            }
        if model_name == "Gradient Boosting":
            return {
                "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "classifier__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "classifier__max_depth": trial.suggest_int("max_depth", 2, 8),
                "classifier__subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "classifier__min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            }
        if model_name == "Extra Trees":
            return {
                "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "classifier__max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                "classifier__min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "classifier__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "classifier__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

    if task_type == config.TASK_REGRESSION:
        if model_name == "Ridge Regression":
            return {"regressor__alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
        if model_name == "SVR":
            return {
                "regressor__C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "regressor__kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                "regressor__gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "regressor__epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True),
            }
        if model_name == "Random Forest Regressor":
            return {
                "regressor__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "regressor__max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                "regressor__min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "regressor__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        if model_name == "XGBoost Regressor":
            return {
                "regressor__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "regressor__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "regressor__max_depth": trial.suggest_int("max_depth", 2, 10),
                "regressor__min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "regressor__subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "regressor__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
        if model_name == "KNN Regressor":
            return {
                "regressor__n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "regressor__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "regressor__metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
            }
        if model_name == "Gradient Boosting Regressor":
            return {
                "regressor__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "regressor__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "regressor__max_depth": trial.suggest_int("max_depth", 2, 8),
                "regressor__subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "regressor__min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            }
        if model_name == "Extra Trees Regressor":
            return {
                "regressor__n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "regressor__max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                "regressor__min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
                "regressor__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }

    raise ValueError(f"No Optuna space defined for {task_type} model: {model_name}")


def _optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    task_type: str,
) -> float:
    params = _build_optuna_params(trial, model_name, task_type)
    cloned = Pipeline(pipeline.steps)
    cloned.set_params(**params)

    scoring = (config.SCORING_METRIC_CLF
               if task_type == config.TASK_CLASSIFICATION
               else config.SCORING_METRIC_REG)

    if task_type == config.TASK_CLASSIFICATION:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    scores = cross_val_score(cloned, X_train, y_train, cv=cv, scoring=scoring, n_jobs=config.N_JOBS)
    return float(scores.mean())


def auto_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = config.OPTUNA_TRIALS,
    task_type: str = config.TASK_CLASSIFICATION,
) -> Tuple[Optional[Pipeline], Optional[str], pd.DataFrame, Dict[str, Pipeline]]:
    """Run Optuna-based hyperparameter search across all registered models."""
    configs = get_model_configs(task_type)
    is_binary = (task_type == config.TASK_CLASSIFICATION and len(np.unique(y_test)) == 2)

    if task_type == config.TASK_CLASSIFICATION:
        cv_folds = min(config.CV_FOLDS, int(y_train.value_counts().min()))
        cv_folds = max(cv_folds, 2)
    else:
        cv_folds = config.CV_FOLDS

    records: List[Dict[str, Any]] = []
    all_pipelines: Dict[str, Pipeline] = {}
    best_score = -np.inf
    best_pipeline: Optional[Pipeline] = None
    best_model_name: Optional[str] = None

    model_names = list(configs.keys())
    total = len(model_names)

    overall_bar = st.progress(0.0, text="Starting AutoTrain…")
    status_box  = st.empty()

    for idx, model_name in enumerate(model_names):
        status_box.info(f"🔧 Tuning **{model_name}** ({idx + 1}/{total})…")
        cfg = configs[model_name]

        try:
            study = optuna.create_study(direction="maximize", study_name=model_name)
            study.optimize(
                lambda trial, mn=model_name, pl=cfg["pipeline"]: _optuna_objective(
                    trial, mn, pl, X_train, y_train, cv_folds, task_type
                ),
                n_trials=n_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            step_prefix = "classifier__" if task_type == config.TASK_CLASSIFICATION else "regressor__"
            prefixed = {
                (k if k.startswith(step_prefix) else f"{step_prefix}{k}"): v
                for k, v in best_params.items()
            }
            final_pipeline = Pipeline(cfg["pipeline"].steps)
            final_pipeline.set_params(**prefixed)
            final_pipeline.fit(X_train, y_train)

            y_pred = final_pipeline.predict(X_test)

            if task_type == config.TASK_CLASSIFICATION:
                y_prob = None
                if hasattr(final_pipeline, "predict_proba"):
                    proba = final_pipeline.predict_proba(X_test)
                    y_prob = proba[:, 1] if is_binary else None
                metrics = compute_metrics(y_test.values, y_pred, y_prob, is_binary)
                score_key = "Accuracy"
            else:
                metrics = compute_regression_metrics(y_test.values, y_pred)
                score_key = "R²"

            records.append({
                "Model":    model_name,
                "CV Score": round(study.best_value, 4),
                **{k: (round(v, 4) if v is not None else None) for k, v in metrics.items()},
            })
            all_pipelines[model_name] = final_pipeline

            if metrics[score_key] > best_score:
                best_score      = metrics[score_key]
                best_pipeline   = final_pipeline
                best_model_name = model_name

        except Exception as exc:
            logger.error("Failed to train %s: %s", model_name, exc)
            st.warning(f"⚠️ {model_name} failed: {exc}")

        overall_bar.progress((idx + 1) / total, text=f"Completed {idx + 1}/{total} models")

    status_box.empty()
    overall_bar.empty()

    results_df = pd.DataFrame(records)
    return best_pipeline, best_model_name, results_df, all_pipelines


def evaluate_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = config.TASK_CLASSIFICATION,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """Train every registered model with GridSearchCV and return fitted pipelines plus metrics."""
    configs_   = get_model_configs(task_type)
    is_binary  = (task_type == config.TASK_CLASSIFICATION and len(np.unique(y_test)) == 2)
    fitted_models: Dict[str, Pipeline] = {}
    records: List[Dict] = []

    for name in configs_:
        try:
            pipeline, _ = train_single_model(name, X_train, y_train, task_type)
            fitted_models[name] = pipeline

            y_pred = pipeline.predict(X_test)

            if task_type == config.TASK_CLASSIFICATION:
                y_prob = None
                if hasattr(pipeline, "predict_proba"):
                    proba  = pipeline.predict_proba(X_test)
                    y_prob = proba[:, 1] if is_binary else None
                metrics = compute_metrics(y_test.values, y_pred, y_prob, is_binary)
            else:
                metrics = compute_regression_metrics(y_test.values, y_pred)

            records.append({"Model": name, **metrics})
        except Exception as exc:
            logger.error("evaluate_all_models – %s failed: %s", name, exc)
            st.warning(f"⚠️ Could not evaluate {name}: {exc}")

    results_df = pd.DataFrame(records).set_index("Model")
    return fitted_models, results_df
