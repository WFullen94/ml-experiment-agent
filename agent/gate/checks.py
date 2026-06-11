from __future__ import annotations

from agent.core.datasets import REGISTRY, get as get_profile
from agent.core.loop import CLASSIC_METHODS, COMPUTE_BUDGET_GPU_MIN
from agent.core.types import GateOutcome, GateResult, Plan

IMBALANCE_THRESHOLD = 5.0  # majority:minority ratio above which accuracy is misleading


# --- check 1: wrong metric for imbalanced data -------------------------------

def check_accuracy_on_imbalanced(plan: Plan) -> GateResult | None:
    """Block plans that use accuracy as the metric on heavily imbalanced datasets."""
    if plan.metric != "accuracy":
        return None
    try:
        profile = get_profile(plan.dataset)
    except KeyError:
        return None  # check 4 will handle unknown datasets
    if profile.task_type != "classification":
        return None
    if profile.imbalance_ratio > IMBALANCE_THRESHOLD:
        return GateResult(
            outcome=GateOutcome.BLOCKED,
            reason=(
                f"accuracy is misleading on '{plan.dataset}' "
                f"({profile.imbalance_ratio:.1f}:1 class imbalance) — "
                f"use auc_pr or f1 instead"
            ),
        )
    return None


# --- check 2: k-fold CV on temporal data -------------------------------------

def check_kfold_on_temporal(plan: Plan) -> GateResult | None:
    """Block k-fold CV on time-ordered datasets — future data leaks into training folds."""
    if plan.cv_strategy != "kfold":
        return None
    try:
        profile = get_profile(plan.dataset)
    except KeyError:
        return None
    if profile.has_time_index:
        return GateResult(
            outcome=GateOutcome.BLOCKED,
            reason=(
                f"k-fold CV on '{plan.dataset}' leaks future data into training folds "
                f"(dataset has a time index) — use walk_forward instead"
            ),
        )
    return None


# --- check 3: compute budget -------------------------------------------------

def check_compute_budget(plan: Plan) -> GateResult | None:
    """Route to scaffold mode if any run uses a method too expensive to execute locally."""
    expensive = [r.method for r in plan.runs if r.method not in CLASSIC_METHODS]
    if not expensive:
        return None
    cost_estimate = 999.0 * len(expensive)
    return GateResult(
        outcome=GateOutcome.SCAFFOLD,
        reason=(
            f"method(s) {expensive} require ~{cost_estimate:.0f} GPU-min — "
            f"budget is {COMPUTE_BUDGET_GPU_MIN} GPU-min; "
            f"returning validated scaffold instead of executing"
        ),
    )


# --- check 4: dataset not registered -----------------------------------------

def check_dataset_available(plan: Plan) -> GateResult | None:
    """Block plans that reference a dataset not in the registry."""
    if plan.dataset not in REGISTRY:
        supported = list(REGISTRY.keys())
        return GateResult(
            outcome=GateOutcome.BLOCKED,
            reason=(
                f"dataset '{plan.dataset}' is not registered; "
                f"supported datasets: {supported}"
            ),
        )
    return None


# --- ordered list of all week-1 checks ---------------------------------------
# Checks run in this order; the first non-None result wins.
# Feasibility checks (3, 4) run before methodology checks (1, 2) so we don't
# waste time validating methodology on a plan that can't run anyway.

ALL_CHECKS = [
    check_dataset_available,   # 4 — fast registry lookup, fail early
    check_compute_budget,      # 3 — route expensive plans to scaffold immediately
    check_accuracy_on_imbalanced,  # 1 — methodology: wrong metric
    check_kfold_on_temporal,   # 2 — methodology: temporal leakage
]
