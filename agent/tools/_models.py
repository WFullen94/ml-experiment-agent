from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Sensible defaults merged with any caller-supplied hyperparams.
DEFAULTS: dict[str, dict] = {
    "logreg":        {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 100, "random_state": 42, "class_weight": "balanced"},
    "gbm":           {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
    "small_mlp":     {"hidden_layer_sizes": (64, 32), "max_iter": 200, "random_state": 42},
    "ridge":         {"alpha": 1.0},
    "rf_regressor":  {"n_estimators": 100, "random_state": 42},
    "gbm_regressor": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
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

SUPPORTED_METHODS = list(_CLASSIFIERS) + list(_REGRESSORS)


def build_model(method: str, task_type: str, hyperparams: dict | None = None):
    """Return a fitted-ready sklearn estimator with hyperparams merged over defaults."""
    hp = {**DEFAULTS.get(method, {}), **(hyperparams or {})}
    builders = _CLASSIFIERS if task_type == "classification" else _REGRESSORS
    if method not in builders:
        raise ValueError(f"unknown method '{method}'; supported: {SUPPORTED_METHODS}")
    return builders[method](**hp), hp
