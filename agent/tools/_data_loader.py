from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Columns held out as targets — shared across all tools.
TARGET_COL: dict[str, str] = {
    "churn": "churned",
    "sales": "revenue",
}

_RNG = np.random.default_rng(42)


def load(dataset: str) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, target_column_name) for a supported dataset."""
    loaders = {"churn": _churn, "sales": _sales, "sst2": _sst2_stub}
    if dataset not in loaders:
        raise KeyError(f"dataset '{dataset}' not registered; supported: {list(loaders)}")
    return loaders[dataset]()


def features_and_target(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience: return (X, y) split."""
    df, target = load(dataset)
    return df.drop(columns=[target]), df[target]


# --- dataset generators -------------------------------------------------------

def _churn() -> tuple[pd.DataFrame, str]:
    X, y = make_classification(
        n_samples=10_000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.92, 0.08],   # ~12:1 imbalance
        flip_y=0.01,
        random_state=42,
    )
    cols = [
        "tenure_months", "monthly_charges", "total_charges", "num_products",
        "has_tech_support", "has_online_backup", "contract_type", "payment_method",
        "num_complaints", "avg_call_duration",
        *[f"feature_{i}" for i in range(10)],
    ]
    df = pd.DataFrame(X, columns=cols)
    df["churned"] = y
    return df, "churned"


def _sales() -> tuple[pd.DataFrame, str]:
    n = 5_000
    rng = _RNG
    dates = pd.date_range("2018-01-01", periods=n, freq="D")

    trend = np.linspace(100, 200, n)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = rng.normal(0, 5, n)
    revenue = trend + seasonality + noise

    df = pd.DataFrame({
        "date": dates,
        "revenue": revenue.round(2),
        "lag_1": np.roll(revenue, 1),
        "lag_7": np.roll(revenue, 7),
        "rolling_mean_30": pd.Series(revenue).rolling(30, min_periods=1).mean().values,
        "month": dates.month,
        "day_of_week": dates.dayofweek,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    })
    # time index — the gate uses this to block k-fold CV
    df = df.set_index("date")
    return df, "revenue"


def _sst2_stub() -> tuple[pd.DataFrame, str]:
    # SST-2 is scaffold-only — this stub exists so inspect_data can report
    # basic stats without loading 67k examples.
    df = pd.DataFrame({
        "sentence": ["[stub — sst2 is scaffold-only]"],
        "label": [0],
    })
    return df, "label"
