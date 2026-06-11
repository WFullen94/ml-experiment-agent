"""
Planner tests.

Unit tests cover message construction and response parsing without any API
calls. The integration test (marked) makes a real API call and is skipped
unless ANTHROPIC_API_KEY is set in the environment.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from agent.core.planner import ApiPlanner, _build_messages, _parse_response
from agent.core.types import (
    ExperimentRun,
    GateOutcome,
    GateResult,
    Plan,
    StepType,
    TrajectoryStep,
)


# --- helpers -----------------------------------------------------------------

def _make_plan_step(plan_dict: dict, step: int = 0) -> TrajectoryStep:
    return TrajectoryStep(
        step=step,
        type=StepType.PLAN,
        input={"goal": "test"},
        output={"plan": plan_dict},
    )


def _make_gate_step(outcome: GateOutcome, reason: str = "", step: int = 1) -> TrajectoryStep:
    return TrajectoryStep(
        step=step,
        type=StepType.GATE_CHECK,
        input={},
        output={},
        gate_result=GateResult(outcome=outcome, reason=reason),
    )


def _mock_response(plan_dict: dict):
    """Minimal mock of an Anthropic API response with a tool-use block."""
    block = SimpleNamespace(
        type="tool_use",
        name="submit_plan",
        input=plan_dict,
    )
    return SimpleNamespace(content=[block])


# --- _build_messages ---------------------------------------------------------

class TestBuildMessages:
    def test_empty_trajectory_is_single_user_message(self):
        msgs = _build_messages("compare models on churn", [])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "compare models on churn"

    def test_previous_blocked_plan_adds_three_messages(self):
        plan_dict = {"dataset": "churn", "runs": [{"method": "logreg"}],
                     "metric": "accuracy", "cv_strategy": "kfold"}
        traj = [
            _make_plan_step(plan_dict, step=0),
            _make_gate_step(GateOutcome.BLOCKED, "accuracy on imbalanced data", step=1),
        ]
        msgs = _build_messages("compare models on churn", traj)
        # user goal + assistant tool call + user tool result
        assert len(msgs) == 3
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"][0]["type"] == "tool_use"
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"][0]["type"] == "tool_result"
        assert "accuracy" in msgs[2]["content"][0]["content"]

    def test_approved_gate_does_not_add_tool_result(self):
        plan_dict = {"dataset": "churn", "runs": [{"method": "logreg"}],
                     "metric": "auc_pr", "cv_strategy": "kfold"}
        traj = [
            _make_plan_step(plan_dict, step=0),
            _make_gate_step(GateOutcome.APPROVED, step=1),
        ]
        msgs = _build_messages("compare models on churn", traj)
        # user goal + assistant tool call only (no tool result for approval)
        assert len(msgs) == 2

    def test_multiple_rejections_produce_full_conversation(self):
        plan = {"dataset": "churn", "runs": [{"method": "logreg"}],
                "metric": "accuracy", "cv_strategy": "kfold"}
        traj = [
            _make_plan_step(plan, step=0),
            _make_gate_step(GateOutcome.BLOCKED, "bad metric", step=1),
            _make_plan_step(plan, step=2),
            _make_gate_step(GateOutcome.BLOCKED, "still bad", step=3),
        ]
        msgs = _build_messages("goal", traj)
        # goal + (tool_call + tool_result) * 2
        assert len(msgs) == 5


# --- _parse_response ---------------------------------------------------------

class TestParseResponse:
    def test_parses_plan_correctly(self):
        resp = _mock_response({
            "dataset":     "churn",
            "runs":        [{"method": "logreg"}, {"method": "random_forest"}],
            "metric":      "auc_pr",
            "cv_strategy": "kfold",
        })
        plan = _parse_response(resp, goal="compare classifiers on churn")
        assert plan.dataset == "churn"
        assert len(plan.runs) == 2
        assert plan.runs[0].method == "logreg"
        assert plan.runs[1].method == "random_forest"
        assert plan.metric == "auc_pr"
        assert plan.cv_strategy == "kfold"
        assert plan.goal == "compare classifiers on churn"

    def test_hyperparams_default_to_empty_dict(self):
        resp = _mock_response({
            "dataset": "churn",
            "runs": [{"method": "logreg"}],
            "metric": "auc_pr",
            "cv_strategy": "kfold",
        })
        plan = _parse_response(resp, "goal")
        assert plan.runs[0].hyperparams == {}

    def test_hyperparams_are_preserved(self):
        resp = _mock_response({
            "dataset": "sst2",
            "runs": [{"method": "lora", "hyperparams": {"rank": 8}}],
            "metric": "accuracy",
            "cv_strategy": "holdout",
        })
        plan = _parse_response(resp, "goal")
        assert plan.runs[0].hyperparams == {"rank": 8}

    def test_missing_tool_call_raises(self):
        bad_resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="oops")])
        with pytest.raises(ValueError, match="submit_plan"):
            _parse_response(bad_resp, "goal")


# --- integration test --------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestApiPlannerIntegration:
    def test_churn_comparison_goal_returns_valid_plan(self):
        planner = ApiPlanner()
        plan = planner(
            "Compare logistic regression vs random forest on the churn dataset.",
            trajectory=[],
        )
        assert isinstance(plan, Plan)
        assert plan.dataset == "churn"
        assert len(plan.runs) >= 2
        assert plan.metric in ("auc_pr", "f1")   # should not choose accuracy

    def test_lora_goal_returns_scaffold_method(self):
        planner = ApiPlanner()
        plan = planner(
            "Compare LoRA rank 8 vs rank 32 fine-tuning on SST-2.",
            trajectory=[],
        )
        assert plan.dataset == "sst2"
        methods = [r.method for r in plan.runs]
        assert all(m in ("lora", "qlora", "full_ft") for m in methods)

    def test_planner_revises_after_rejection(self):
        from agent.core.types import GateOutcome, GateResult, StepType, TrajectoryStep
        bad_plan_dict = {
            "dataset": "churn", "runs": [{"method": "logreg"}],
            "metric": "accuracy", "cv_strategy": "kfold",
        }
        traj = [
            TrajectoryStep(0, StepType.PLAN, {"goal": "test"}, {"plan": bad_plan_dict}),
            TrajectoryStep(1, StepType.GATE_CHECK, {}, {},
                           GateResult(GateOutcome.BLOCKED,
                                      "accuracy is misleading on churn (12:1 imbalance)")),
        ]
        planner = ApiPlanner()
        plan = planner("Compare logistic regression on churn.", traj)
        assert plan.metric != "accuracy"
