from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from agent.core.datasets import get as get_profile
from agent.tools._data_loader import features_and_target

# Sensible defaults merged with any caller-supplied hyperparams.
_DEFAULTS: dict[str, dict] = {
    "logreg":         {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    "random_forest":  {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"},
    "gbm":            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
    "small_mlp":      {"hidden_layer_sizes": (64, 32), "max_iter": 200, "random_state": 42},
    # regression variants
    "ridge":          {"alpha": 1.0},
    "rf_regressor":   {"n_estimators": 100, "random_state": 42},
    "gbm_regressor":  {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
}

_CLASSIFIERS = {
    "logreg":        LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gbm":           GradientBoostingClassifier,
    "small_mlp":     MLPClassifier,
}

_REGRESSORS = {
    "ridge":         Ridge,
    "rf_regressor":  RandomForestRegressor,
    "gbm_regressor": GradientBoostingRegressor,
    "small_mlp":     MLPRegressor,
}


def train_model(dataset: str, method: str, hyperparams: dict | None = None) -> dict:
    """
    Fit one model on an 80/20 train/test split and return performance scores.

    Returns
    -------
    method, dataset, hyperparams, n_train, n_test, scores
    scores keys: accuracy + auc_pr + f1  (classification)
                 mae + rmse + r2         (regression)
    """
    profile = get_profile(dataset)
    X, y = features_and_target(dataset)

    hp = {**_DEFAULTS.get(method, {}), **(hyperparams or {})}
    model = _build(method, profile.task_type, hp)

    if profile.task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        model.fit(X_train, y_train)
        scores = _clf_scores(model, X_test, y_test)
    else:
        # Temporal data: hold out the last 20% by row order — no shuffling.
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model.fit(X_train, y_train)
        scores = _reg_scores(model, X_test, y_test)

    return {
        "method": method,
        "dataset": dataset,
        "hyperparams": hp,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scores": scores,
    }


# --- helpers -----------------------------------------------------------------

def _build(method: str, task_type: str, hp: dict):
    builders = _CLASSIFIERS if task_type == "classification" else _REGRESSORS
    if method not in builders:
        available = list(_CLASSIFIERS) + list(_REGRESSORS)
        raise ValueError(f"unknown method '{method}'; supported: {available}")
    return builders[method](**hp)


def _clf_scores(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "auc_pr":   round(float(average_precision_score(y_test, y_prob)), 4),
        "f1":       round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }


def _reg_scores(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }
