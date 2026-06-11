from __future__ import annotations

import hashlib
import json
from typing import Callable, Protocol

from .exceptions import LoopDetectedError, MaxRevisionsExceededError
from .types import GateOutcome, GateResult, Plan, StepType, Trajectory, TrajectoryStep

MAX_REVISIONS = 3

# Methods that run in seconds on CPU — anything else is "expensive" and gets scaffolded.
CLASSIC_METHODS = {"logreg", "random_forest", "gbm", "small_mlp",
                   "ridge", "rf_regressor", "gbm_regressor"}
COMPUTE_BUDGET_GPU_MIN = 1.0


class Planner(Protocol):
    def __call__(self, goal: str, trajectory: Trajectory) -> Plan: ...


class Gate(Protocol):
    def __call__(self, plan: Plan) -> GateResult: ...


class ScaffoldGenerator(Protocol):
    def __call__(self, plan: Plan) -> dict: ...


class AgentLoop:
    def __init__(
        self,
        planner: Planner,
        gate: Gate,
        toolbox: dict[str, Callable],
        scaffold_generator: ScaffoldGenerator,
        confirm: Callable[[Plan, float], bool] | None = None,
        max_revisions: int = MAX_REVISIONS,
    ):
        self.planner = planner
        self.gate = gate
        self.toolbox = toolbox
        self.scaffold_generator = scaffold_generator
        self.confirm = confirm if confirm is not None else _cli_confirm
        self.max_revisions = max_revisions

    def run(self, goal: str) -> Trajectory:
        trajectory: Trajectory = []
        seen_calls: set[str] = set()
        revisions = 0

        while True:
            plan = self.planner(goal, trajectory)
            _record(trajectory, StepType.PLAN, {"goal": goal}, {"plan": plan.to_dict()})

            gate_result = self.gate(plan)
            _record(trajectory, StepType.GATE_CHECK, {"plan": plan.to_dict()}, {}, gate_result)

            if gate_result.outcome == GateOutcome.BLOCKED:
                revisions += 1
                if revisions > self.max_revisions:
                    _record(trajectory, StepType.RESULT, {}, {
                        "status": "failed",
                        "error": f"gate blocked plan {self.max_revisions} times",
                        "last_reason": gate_result.reason,
                    })
                    raise MaxRevisionsExceededError(gate_result.reason)
                # planner will see the rejection reason via trajectory on next iteration
                continue

            if gate_result.outcome == GateOutcome.SCAFFOLD:
                return self._scaffold(plan, trajectory)

            # APPROVED — human confirms before spending any compute
            cost = _estimate_cost(plan)
            if not self.confirm(plan, cost):
                _record(trajectory, StepType.RESULT, {}, {"status": "cancelled_by_user"})
                return trajectory

            return self._execute(plan, trajectory, seen_calls)

    def _execute(self, plan: Plan, trajectory: Trajectory, seen_calls: set[str]) -> Trajectory:
        def call(tool_name: str, **kwargs) -> dict:
            sig = _call_sig(tool_name, kwargs)
            if sig in seen_calls:
                raise LoopDetectedError(f"loop: '{tool_name}' called with identical args twice")
            seen_calls.add(sig)
            if tool_name not in self.toolbox:
                raise KeyError(f"tool '{tool_name}' not registered in toolbox")
            result = self.toolbox[tool_name](**kwargs)
            _record(trajectory, StepType.TOOL_CALL, {"tool": tool_name, "args": kwargs}, result)
            return result

        call("inspect_data", dataset=plan.dataset)

        for run in plan.runs:
            call("train_model", dataset=plan.dataset, method=run.method, hyperparams=run.hyperparams)

        cv_result = call("cross_validate",
            dataset=plan.dataset,
            runs=[r.method for r in plan.runs],
            strategy=plan.cv_strategy,
            metric=plan.metric,
        )

        if len(plan.runs) > 1:
            fold_scores = {m: cv_result["results"][m]["fold_scores"] for m in cv_result["results"]}
            call("compare_models", fold_scores=fold_scores, metric=plan.metric)

        for run in plan.runs:
            call("feature_importance", dataset=plan.dataset, method=run.method, hyperparams=run.hyperparams)

        _record(trajectory, StepType.RESULT, {}, {"status": "complete"})
        return trajectory

    def _scaffold(self, plan: Plan, trajectory: Trajectory) -> Trajectory:
        output = self.scaffold_generator(plan)
        _record(trajectory, StepType.SCAFFOLD, {"plan": plan.to_dict()}, output)
        _record(trajectory, StepType.RESULT, {}, {"status": "scaffold_only"})
        return trajectory


# --- helpers -----------------------------------------------------------------

def _record(
    trajectory: Trajectory,
    step_type: StepType,
    inp: dict,
    out: dict,
    gate_result: GateResult | None = None,
) -> TrajectoryStep:
    step = TrajectoryStep(
        step=len(trajectory),
        type=step_type,
        input=inp,
        output=out,
        gate_result=gate_result,
    )
    trajectory.append(step)
    return step


def _call_sig(tool_name: str, args: dict) -> str:
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _estimate_cost(plan: Plan) -> float:
    """Returns estimated GPU-minutes. Anything not in CLASSIC_METHODS is expensive."""
    if all(r.method in CLASSIC_METHODS for r in plan.runs):
        return 0.1 * len(plan.runs)
    return 999.0



def _cli_confirm(plan: Plan, estimated_cost: float) -> bool:
    print("\nProposed plan:")
    print(f"  dataset:  {plan.dataset}")
    print(f"  methods:  {[r.method for r in plan.runs]}")
    print(f"  metric:   {plan.metric}")
    print(f"  cv:       {plan.cv_strategy}")
    print(f"  cost est: ~{estimated_cost:.1f} GPU-min")
    return input("\nApprove? [y/n]: ").strip().lower() == "y"
