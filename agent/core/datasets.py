from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    task_type: str          # "classification" | "regression"
    imbalance_ratio: float  # majority:minority class ratio; 1.0 if regression
    has_time_index: bool
    n_rows: int
    available: bool = True


# Profiles for the three week-1 supported datasets.
# Gate checks read from here — no need to load data to check feasibility.
REGISTRY: dict[str, DatasetProfile] = {
    "churn": DatasetProfile(
        name="churn",
        task_type="classification",
        imbalance_ratio=12.0,   # ~92:8 non-churn:churn typical
        has_time_index=False,
        n_rows=10_000,
    ),
    "sales": DatasetProfile(
        name="sales",
        task_type="regression",
        imbalance_ratio=1.0,
        has_time_index=True,    # monthly time-series → k-fold is leakage
        n_rows=5_000,
    ),
    "sst2": DatasetProfile(
        name="sst2",
        task_type="classification",
        imbalance_ratio=1.1,
        has_time_index=False,
        n_rows=67_349,
    ),
}


def get(name: str) -> DatasetProfile:
    if name not in REGISTRY:
        raise KeyError(f"dataset '{name}' not registered; supported: {list(REGISTRY)}")
    return REGISTRY[name]


def is_available(name: str) -> bool:
    return name in REGISTRY and REGISTRY[name].available
