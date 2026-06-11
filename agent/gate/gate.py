from __future__ import annotations

from agent.core.types import GateOutcome, GateResult, Plan
from agent.gate.checks import ALL_CHECKS


class Gate:
    """
    Runs each check in ALL_CHECKS order and returns the first non-None result.
    Returns APPROVED if all checks pass.

    Checks run in priority order:
      1. dataset_available  — fail fast on unknown datasets
      2. compute_budget     — route expensive plans to scaffold immediately
      3. accuracy_on_imbalanced — methodology: wrong metric
      4. kfold_on_temporal  — methodology: temporal leakage
    """

    def __call__(self, plan: Plan) -> GateResult:
        for check in ALL_CHECKS:
            result = check(plan)
            if result is not None:
                return result
        return GateResult(outcome=GateOutcome.APPROVED)

    def explain(self, plan: Plan) -> list[dict]:
        """Run all checks and return a full report — used for debugging and the eval."""
        report = []
        for check in ALL_CHECKS:
            result = check(plan)
            report.append({
                "check": check.__name__,
                "outcome": result.outcome.value if result else "passed",
                "reason": result.reason if result else "",
            })
        return report
