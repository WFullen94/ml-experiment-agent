from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)

from agent.core.datasets import get as get_profile
from agent.tools._data_loader import features_and_target
from agent.tools._models import build_model

N_FOLDS = 5

# sklearn scorer strings — negated scorers are flipped back to positive.
_SCORERS: dict[str, str] = {
    "accuracy":  "accuracy",
    "auc_pr":    "average_precision",
    "f1":        "f1",
    "mae":       "neg_mean_absolute_error",
    "rmse":      "neg_root_mean_squared_error",
    "r2":        "r2",
}

_NEGATED = {"neg_mean_absolute_error", "neg_root_mean_squared_error"}


def cross_validate(
    dataset: str,
    runs: list[str],
    strategy: str,
    metric: str,
) -> dict:
    """
    Run k-fold or walk-forward CV for each method and return fold-level scores.

    Parameters
    ----------
    dataset   : registered dataset name
    runs      : list of method names (e.g. ["logreg", "random_forest"])
    strategy  : "kfold" | "walk_forward"
    metric    : "accuracy" | "auc_pr" | "f1" | "mae" | "rmse" | "r2"

    Returns
    -------
    {
        "results": {
            "<method>": {"mean": float, "std": float, "fold_scores": list[float]}
        },
        "metric": str,
        "strategy": str,
        "n_folds": int,
    }
    """
    if metric not in _SCORERS:
        raise ValueError(f"unknown metric '{metric}'; supported: {list(_SCORERS)}")

    profile = get_profile(dataset)
    X, y = features_and_target(dataset)
    cv = _get_cv(strategy, profile.task_type)
    scorer = _SCORERS[metric]

    results: dict[str, dict] = {}
    for method in runs:
        model, _ = build_model(method, profile.task_type)
        raw = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        scores = -raw if scorer in _NEGATED else raw
        results[method] = {
            "mean":        round(float(scores.mean()), 4),
            "std":         round(float(scores.std()), 4),
            "fold_scores": [round(float(s), 4) for s in scores],
        }

    return {
        "results":  results,
        "metric":   metric,
        "strategy": strategy,
        "n_folds":  N_FOLDS,
    }


# --- helpers -----------------------------------------------------------------

def _get_cv(strategy: str, task_type: str):
    if strategy == "walk_forward":
        return TimeSeriesSplit(n_splits=N_FOLDS)
    if strategy == "kfold":
        if task_type == "classification":
            return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        return KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    raise ValueError(f"unknown cv strategy '{strategy}'; supported: kfold, walk_forward")
