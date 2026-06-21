"""A deterministic stand-in for the LLM ApiPlanner.

Implements the same Planner protocol — ``__call__(goal, trajectory) -> Plan`` —
but chooses plans from fixed rules instead of calling Claude. Crucially, it
reads the trajectory for prior gate rejections and *revises* exactly like the
real planner does, so the real Gate's block→revise loop is genuinely exercised.

This lets agent-eval evaluate the real agent offline and reproducibly.
"""
from __future__ import annotations

from agent.core.types import ExperimentRun, GateOutcome, Plan, StepType, Trajectory


def _count_blocks(trajectory: Trajectory) -> int:
    return sum(
        1 for s in trajectory
        if s.type == StepType.GATE_CHECK
        and s.gate_result is not None
        and s.gate_result.outcome == GateOutcome.BLOCKED
    )


class ScriptedPlanner:
    """Deterministic planner that mirrors the demo flows.

    - churn comparison: first proposes `accuracy` (the gate blocks it on the
      imbalanced data), then revises to `auc_pr` on the next attempt.
    - sales forecast: first proposes `kfold` (blocked — temporal leakage), then
      revises to `walk_forward`.
    - LoRA on SST-2: proposes a LoRA rank comparison, which the compute-budget
      check routes to SCAFFOLD (no revision needed).
    """

    def __call__(self, goal: str, trajectory: Trajectory) -> Plan:
        g = goal.lower()
        attempt = _count_blocks(trajectory)

        if "churn" in g:
            metric = "auc_pr" if attempt >= 1 else "accuracy"
            return Plan(
                dataset="churn",
                runs=[ExperimentRun("logreg"), ExperimentRun("random_forest")],
                metric=metric,
                cv_strategy="kfold",
                goal=goal,
            )

        if "sales" in g or "revenue" in g or "forecast" in g:
            cv = "walk_forward" if attempt >= 1 else "kfold"
            return Plan(
                dataset="sales",
                runs=[ExperimentRun("ridge"), ExperimentRun("rf_regressor")],
                metric="mae",
                cv_strategy=cv,
                goal=goal,
            )

        # LoRA / SST-2 → scaffold path.
        return Plan(
            dataset="sst2",
            runs=[
                ExperimentRun("lora", {"rank": 8}),
                ExperimentRun("lora", {"rank": 32}),
            ],
            metric="accuracy",
            cv_strategy="holdout",
            goal=goal,
        )
