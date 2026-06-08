from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from agent.core.datasets import get as get_profile
from agent.tools._data_loader import features_and_target
from agent.tools._models import build_model


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
    model, hp = build_model(method, profile.task_type, hyperparams)

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
