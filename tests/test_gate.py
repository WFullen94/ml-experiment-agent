import pytest

from agent.core.types import ExperimentRun, GateOutcome, Plan
from agent.gate.gate import Gate
from agent.gate.checks import (
    check_accuracy_on_imbalanced,
    check_compute_budget,
    check_dataset_available,
    check_kfold_on_temporal,
)

gate = Gate()


def _plan(
    dataset="churn",
    methods=("logreg", "random_forest"),
    metric="auc_pr",
    cv="kfold",
):
    return Plan(
        dataset=dataset,
        runs=[ExperimentRun(method=m) for m in methods],
        metric=metric,
        cv_strategy=cv,
        goal="test",
    )


# --- full gate: APPROVED path ------------------------------------------------

class TestApproved:
    def test_sound_plan_is_approved(self):
        plan = _plan(metric="auc_pr", cv="kfold")
        result = gate(plan)
        assert result.outcome == GateOutcome.APPROVED

    def test_f1_metric_passes(self):
        result = gate(_plan(metric="f1"))
        assert result.outcome == GateOutcome.APPROVED

    def test_walk_forward_on_sales_passes(self):
        result = gate(_plan(dataset="sales", metric="mae", cv="walk_forward",
                            methods=("rf_regressor",)))
        assert result.outcome == GateOutcome.APPROVED


# --- check 1: accuracy on imbalanced data ------------------------------------

class TestAccuracyOnImbalanced:
    def test_accuracy_on_churn_is_blocked(self):
        result = check_accuracy_on_imbalanced(_plan(metric="accuracy"))
        assert result.outcome == GateOutcome.BLOCKED
        assert "accuracy" in result.reason
        assert "imbalance" in result.reason

    def test_auc_pr_on_churn_passes(self):
        assert check_accuracy_on_imbalanced(_plan(metric="auc_pr")) is None

    def test_accuracy_on_balanced_dataset_passes(self):
        # sst2 has imbalance_ratio ~1.1 — below threshold
        assert check_accuracy_on_imbalanced(_plan(dataset="sst2", metric="accuracy")) is None

    def test_accuracy_on_regression_passes(self):
        # sales is regression — accuracy check doesn't apply
        assert check_accuracy_on_imbalanced(
            _plan(dataset="sales", metric="accuracy", cv="walk_forward",
                  methods=("rf_regressor",))
        ) is None

    def test_gate_blocks_accuracy_on_churn_end_to_end(self):
        result = gate(_plan(metric="accuracy"))
        assert result.outcome == GateOutcome.BLOCKED


# --- check 2: k-fold on temporal data ----------------------------------------

class TestKFoldOnTemporal:
    def test_kfold_on_sales_is_blocked(self):
        result = check_kfold_on_temporal(
            _plan(dataset="sales", cv="kfold", metric="mae", methods=("rf_regressor",))
        )
        assert result.outcome == GateOutcome.BLOCKED
        assert "time" in result.reason.lower()

    def test_walk_forward_on_sales_passes(self):
        assert check_kfold_on_temporal(
            _plan(dataset="sales", cv="walk_forward", metric="mae",
                  methods=("rf_regressor",))
        ) is None

    def test_kfold_on_non_temporal_passes(self):
        assert check_kfold_on_temporal(_plan(dataset="churn", cv="kfold")) is None

    def test_gate_blocks_kfold_on_sales_end_to_end(self):
        result = gate(_plan(dataset="sales", cv="kfold", metric="mae",
                            methods=("rf_regressor",)))
        assert result.outcome == GateOutcome.BLOCKED


# --- check 3: compute budget -------------------------------------------------

class TestComputeBudget:
    def test_lora_routes_to_scaffold(self):
        result = check_compute_budget(_plan(dataset="sst2", methods=("lora",)))
        assert result.outcome == GateOutcome.SCAFFOLD
        assert "scaffold" in result.reason.lower()

    def test_classic_methods_pass(self):
        assert check_compute_budget(_plan(methods=("logreg", "random_forest"))) is None

    def test_mixed_plan_with_expensive_method_scaffolds(self):
        # even one expensive method triggers scaffold
        result = check_compute_budget(
            _plan(methods=("logreg", "lora"))
        )
        assert result.outcome == GateOutcome.SCAFFOLD

    def test_gate_routes_lora_to_scaffold_end_to_end(self):
        result = gate(_plan(dataset="sst2", methods=("lora",), metric="accuracy"))
        assert result.outcome == GateOutcome.SCAFFOLD


# --- check 4: dataset not registered -----------------------------------------

class TestDatasetAvailable:
    def test_unknown_dataset_is_blocked(self):
        result = check_dataset_available(_plan(dataset="titanic"))
        assert result.outcome == GateOutcome.BLOCKED
        assert "titanic" in result.reason

    def test_registered_datasets_pass(self):
        for ds in ("churn", "sales", "sst2"):
            assert check_dataset_available(_plan(dataset=ds)) is None

    def test_gate_blocks_unknown_dataset_end_to_end(self):
        result = gate(_plan(dataset="titanic"))
        assert result.outcome == GateOutcome.BLOCKED


# --- check priority order ----------------------------------------------------

class TestCheckOrdering:
    def test_dataset_check_fires_before_methodology_check(self):
        # accuracy on unknown dataset — should get BLOCKED for dataset, not metric
        plan = _plan(dataset="titanic", metric="accuracy")
        result = gate(plan)
        assert result.outcome == GateOutcome.BLOCKED
        assert "titanic" in result.reason   # dataset reason, not metric reason

    def test_compute_budget_fires_before_methodology_check(self):
        # lora with accuracy metric — should get SCAFFOLD (budget), not BLOCKED (metric)
        plan = _plan(dataset="sst2", methods=("lora",), metric="accuracy")
        result = gate(plan)
        assert result.outcome == GateOutcome.SCAFFOLD


# --- explain method ----------------------------------------------------------

class TestExplain:
    def test_explain_returns_one_entry_per_check(self):
        from agent.gate.checks import ALL_CHECKS
        report = gate.explain(_plan())
        assert len(report) == len(ALL_CHECKS)

    def test_explain_shows_passed_for_sound_plan(self):
        report = gate.explain(_plan(metric="auc_pr"))
        outcomes = [r["outcome"] for r in report]
        assert all(o == "passed" for o in outcomes)

    def test_explain_shows_blocked_check_for_bad_metric(self):
        report = gate.explain(_plan(metric="accuracy"))
        blocked = [r for r in report if r["outcome"] == "blocked"]
        assert len(blocked) == 1
        assert "accuracy" in blocked[0]["check"]
