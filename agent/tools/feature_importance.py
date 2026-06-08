from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance

from agent.core.datasets import get as get_profile
from agent.tools._data_loader import features_and_target
from agent.tools._models import build_model

TOP_N = 10


def feature_importance(
    dataset: str,
    method: str,
    hyperparams: dict | None = None,
) -> dict:
    """
    Refit a model on the full dataset and return ranked feature importances.

    Uses native importances for tree models and logreg (fast).
    Falls back to permutation importance for MLP (model-agnostic).

    Returns
    -------
    {
        "method": str,
        "dataset": str,
        "importances": [{"feature": str, "importance": float}, ...],  # top-N, descending
        "n_features": int,
        "importance_type": "native" | "permutation",
    }
    """
    profile = get_profile(dataset)
    X, y = features_and_target(dataset)
    feature_names = list(X.columns)

    model, _ = build_model(method, profile.task_type, hyperparams)
    model.fit(X, y)

    scores, importance_type = _extract(model, X, y)

    ranked = sorted(
        zip(feature_names, scores),
        key=lambda t: t[1],
        reverse=True,
    )[:TOP_N]

    return {
        "method":          method,
        "dataset":         dataset,
        "importances":     [{"feature": f, "importance": round(float(s), 4)} for f, s in ranked],
        "n_features":      len(feature_names),
        "importance_type": importance_type,
    }


# --- helpers -----------------------------------------------------------------

def _extract(model, X, y) -> tuple[np.ndarray, str]:
    # Tree-based models expose feature_importances_ natively.
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_, "native"

    # Logistic regression: use absolute coefficient values.
    if hasattr(model, "coef_"):
        coef = model.coef_
        scores = np.abs(coef[0] if coef.ndim > 1 else coef)
        return scores, "native"

    # MLP and anything else: permutation importance on a held-out 20%.
    split = int(len(X) * 0.8)
    X_val, y_val = X.iloc[split:], y.iloc[split:]
    result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
    return np.abs(result.importances_mean), "permutation"
