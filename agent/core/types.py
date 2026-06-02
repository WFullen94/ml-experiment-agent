from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(str, Enum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    GATE_CHECK = "gate_check"
    SCAFFOLD = "scaffold"
    RESULT = "result"


class GateOutcome(str, Enum):
    APPROVED = "approved"
    BLOCKED = "blocked"
    SCAFFOLD = "scaffold"


@dataclass
class GateResult:
    outcome: GateOutcome
    reason: str = ""


@dataclass
class ExperimentRun:
    method: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    dataset: str
    runs: list[ExperimentRun]
    metric: str
    cv_strategy: str
    goal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "dataset": self.dataset,
            "runs": [{"method": r.method, "hyperparams": r.hyperparams} for r in self.runs],
            "metric": self.metric,
            "cv_strategy": self.cv_strategy,
        }


@dataclass
class TrajectoryStep:
    step: int
    type: StepType
    input: dict[str, Any]
    output: dict[str, Any]
    gate_result: GateResult | None = None
    timestamp: float = field(default_factory=time.time)


Trajectory = list[TrajectoryStep]
