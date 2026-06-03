from __future__ import annotations

import pandas as pd

from agent.core.datasets import get as get_profile
from agent.tools._data_loader import load


def inspect_data(dataset: str) -> dict:
    """
    Return a profile of the dataset used by the gate and the planner.

    Keys
    ----
    shape           (n_rows, n_cols including target)
    n_features      columns excluding target
    task_type       "classification" | "regression"
    has_time_index  True if the index is a DatetimeIndex
    dtypes          {col: dtype_string}
    missingness     {col: fraction_missing}
    class_balance   {label: fraction}  — empty for regression
    imbalance_ratio majority/minority class ratio; 1.0 for regression
    """
    profile = get_profile(dataset)
    df, target_col = load(dataset)

    has_time_index = isinstance(df.index, pd.DatetimeIndex)
    all_cols = list(df.columns)
    feature_cols = [c for c in all_cols if c != target_col]

    result: dict = {
        "shape": df.shape,
        "n_features": len(feature_cols),
        "task_type": profile.task_type,
        "has_time_index": has_time_index,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missingness": df.isnull().mean().round(4).to_dict(),
    }

    if profile.task_type == "classification":
        counts = df[target_col].value_counts(normalize=True).sort_index()
        result["class_balance"] = counts.round(4).to_dict()
        minority_frac = counts.min()
        majority_frac = counts.max()
        result["imbalance_ratio"] = (
            round(majority_frac / minority_frac, 2) if minority_frac > 0 else float("inf")
        )
    else:
        result["class_balance"] = {}
        result["imbalance_ratio"] = 1.0

    return result
