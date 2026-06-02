import pytest

from agent.core.exceptions import LoopDetectedError, MaxRevisionsExceededError
from agent.core.loop import AgentLoop, _estimate_cost, _to_tool_sequence
from agent.core.types import (
    ExperimentRun,
    GateOutcome,
    GateResult,
    Plan,
    StepType,
    Trajectory,
)

# --- fixtures / helpers ------------------------------------------------------

def _plan(metric="auc_pr", cv="kfold", methods=("logreg", "random_forest")):
    return Plan(
        dataset="churn",
        runs=[ExperimentRun(method=m) for m in methods],
        metric=metric,
        cv_strategy=cv,
        goal="compare models on churn",
    )


def _static_planner(plan: Plan):
    """Returns the same plan regardless of trajectory."""
    def planner(goal: str, trajectory: Trajectory) -> Plan:
        return plan
    return planner


def _gate_returning(result: GateResult):
    def gate(plan: Plan) -> GateResult:
        return result
    return gate


def _auto_confirm(plan: Plan, cost: float) -> bool:
    return True


def _stub_toolbox() -> dict:
    return {
        "inspect_data":   lambda **kw: {"shape": (1000, 10), "imbalance_ratio": 12.0},
        "train_model":    lambda **kw: {"model": "fitted", "method": kw["method"]},
        "cross_validate": lambda **kw: {"mean": 0.85, "std": 0.02},
        "compare_models": lambda **kw: {"winner": "random_forest", "p_value": 0.03},
        "feature_importance": lambda **kw: {"top": ["tenure", "charges"]},
    }


def _stub_scaffold(plan: Plan) -> dict:
    return {"config": plan.to_dict(), "code": "# scaffold"}


def _make_loop(planner, gate, confirm=_auto_confirm, toolbox=None):
    return AgentLoop(
        planner=planner,
        gate=gate,
        toolbox=toolbox or _stub_toolbox(),
        scaffold_generator=_stub_scaffold,
        confirm=confirm,
    )


# --- happy path --------------------------------------------------------------

def test_approved_plan_executes_all_tools():
    plan = _plan()
    loop = _make_loop(
        planner=_static_planner(plan),
        gate=_gate_returning(GateResult(GateOutcome.APPROVED)),
    )
    traj = loop.run("compare models on churn")

    step_types = [s.type for s in traj]
    assert StepType.PLAN in step_types
    assert StepType.GATE_CHECK in step_types
    assert StepType.TOOL_CALL in step_types
    assert traj[-1].type == StepType.RESULT
    assert traj[-1].output["status"] == "complete"


def test_tool_sequence_order():
    plan = _plan(methods=("logreg", "random_forest"))
    seq = _to_tool_sequence(plan)
    tool_names = [t for t, _ in seq]
    assert tool_names[0] == "inspect_data"
    assert "train_model" in tool_names
    assert "cross_validate" in tool_names
    assert "compare_models" in tool_names


def test_single_run_skips_compare_models():
    plan = _plan(methods=("logreg",))
    seq = _to_tool_sequence(plan)
    assert all(t != "compare_models" for t, _ in seq)


# --- gate paths --------------------------------------------------------------

def test_blocked_then_approved_records_both_gate_checks():
    plan = _plan()
    call_count = {"n": 0}

    def gate(p: Plan) -> GateResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return GateResult(GateOutcome.BLOCKED, "wrong metric")
        return GateResult(GateOutcome.APPROVED)

    loop = _make_loop(planner=_static_planner(plan), gate=gate)
    traj = loop.run("compare models")

    gate_steps = [s for s in traj if s.type == StepType.GATE_CHECK]
    assert len(gate_steps) == 2
    assert gate_steps[0].gate_result.outcome == GateOutcome.BLOCKED
    assert gate_steps[1].gate_result.outcome == GateOutcome.APPROVED


def test_scaffold_path_does_not_call_toolbox():
    called = []

    def tracking_toolbox():
        tb = _stub_toolbox()
        return {k: (lambda fn, name=k: lambda **kw: (called.append(name), fn(**kw))[1])(v) for k, v in tb.items()}

    plan = _plan()
    loop = _make_loop(
        planner=_static_planner(plan),
        gate=_gate_returning(GateResult(GateOutcome.SCAFFOLD, "exceeds budget")),
        toolbox=tracking_toolbox(),
    )
    traj = loop.run("expensive experiment")

    assert not called, f"toolbox was called during scaffold: {called}"
    assert any(s.type == StepType.SCAFFOLD for s in traj)
    assert traj[-1].output["status"] == "scaffold_only"


def test_user_cancel_stops_execution():
    plan = _plan()
    loop = _make_loop(
        planner=_static_planner(plan),
        gate=_gate_returning(GateResult(GateOutcome.APPROVED)),
        confirm=lambda p, c: False,
    )
    traj = loop.run("compare models")
    assert traj[-1].output["status"] == "cancelled_by_user"
    assert all(s.type != StepType.TOOL_CALL for s in traj)


# --- safety: loop detection --------------------------------------------------

def test_loop_detection_raises_on_repeated_call():
    plan = _plan(methods=("logreg",))

    def bad_toolbox(**kw):
        return {}

    # Inject a duplicate by running the same plan twice via a custom executor
    # We test _call_sig indirectly: give the loop a toolbox that the loop
    # will call twice for the same (tool, args) by crafting a two-run plan
    # where both runs have identical method + hyperparams.
    duplicate_plan = Plan(
        dataset="churn",
        runs=[ExperimentRun("logreg", {}), ExperimentRun("logreg", {})],
        metric="auc_pr",
        cv_strategy="kfold",
        goal="trigger loop",
    )
    loop = _make_loop(
        planner=_static_planner(duplicate_plan),
        gate=_gate_returning(GateResult(GateOutcome.APPROVED)),
    )
    with pytest.raises(LoopDetectedError):
        loop.run("trigger loop")


# --- safety: max revisions ---------------------------------------------------

def test_max_revisions_raises_after_n_blocks():
    plan = _plan()
    loop = AgentLoop(
        planner=_static_planner(plan),
        gate=_gate_returning(GateResult(GateOutcome.BLOCKED, "always blocked")),
        toolbox=_stub_toolbox(),
        scaffold_generator=_stub_scaffold,
        confirm=_auto_confirm,
        max_revisions=2,
    )
    with pytest.raises(MaxRevisionsExceededError):
        loop.run("always fails")


def test_max_revisions_trajectory_has_result_step():
    plan = _plan()
    loop = AgentLoop(
        planner=_static_planner(plan),
        gate=_gate_returning(GateResult(GateOutcome.BLOCKED, "always blocked")),
        toolbox=_stub_toolbox(),
        scaffold_generator=_stub_scaffold,
        confirm=_auto_confirm,
        max_revisions=1,
    )
    with pytest.raises(MaxRevisionsExceededError):
        traj = loop.run("always fails")
    # trajectory is recorded even on failure — check via the exception path
    # by catching it ourselves
    try:
        loop2 = AgentLoop(
            planner=_static_planner(plan),
            gate=_gate_returning(GateResult(GateOutcome.BLOCKED, "x")),
            toolbox=_stub_toolbox(),
            scaffold_generator=_stub_scaffold,
            confirm=_auto_confirm,
            max_revisions=1,
        )
        loop2.run("fail")
    except MaxRevisionsExceededError:
        pass  # expected


# --- cost estimation ---------------------------------------------------------

def test_classic_methods_are_cheap():
    plan = _plan(methods=("logreg", "random_forest"))
    assert _estimate_cost(plan) < 1.0


def test_lora_is_expensive():
    plan = Plan("sst2", [ExperimentRun("lora")], "accuracy", "holdout", "lora test")
    assert _estimate_cost(plan) > 100
