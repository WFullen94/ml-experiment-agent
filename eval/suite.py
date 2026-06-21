"""agent-eval suite + real-agent factory for ml-experiment-agent."""
from __future__ import annotations

from agent_eval import EvalSuite, EvalCase, MlExperimentAdapter
from agent_eval.assertions import (
    ToolCalledAssertion,
    ToolSequenceAssertion,
    ToolNotCalledAssertion,
    NoLoopAssertion,
    StepCountAssertion,
    TraceConstraintAssertion,
)

from agent.core.loop import AgentLoop
from agent.gate.gate import Gate
from agent.scaffold.generator import ScaffoldGenerator
from agent.tools.compare_models import compare_models
from agent.tools.cross_validate import cross_validate
from agent.tools.feature_importance import feature_importance
from agent.tools.inspect_data import inspect_data
from agent.tools.train_model import train_model

from .scripted_planner import ScriptedPlanner

TOOLBOX = {
    "inspect_data": inspect_data,
    "train_model": train_model,
    "cross_validate": cross_validate,
    "compare_models": compare_models,
    "feature_importance": feature_importance,
}


def build_real_adapter() -> MlExperimentAdapter:
    """Wire the REAL AgentLoop (real gate + real sklearn tools) behind a
    ScriptedPlanner, and adapt it for agent-eval. No API key required."""
    loop = AgentLoop(
        planner=ScriptedPlanner(),
        gate=Gate(),
        toolbox=TOOLBOX,
        scaffold_generator=ScaffoldGenerator(),
        confirm=lambda plan, cost: True,  # auto-approve
    )
    return MlExperimentAdapter(run_fn=lambda goal: loop.run(goal))


# --- predicates for gate-reliability checks (JSON can't express these) --------

def _gate_blocked(trace) -> bool:
    return any(e.type == "gate_check" and e.content.get("outcome") == "blocked" for e in trace)


def _gate_scaffolded(trace) -> bool:
    return any(e.type == "gate_check" and e.content.get("outcome") == "scaffold" for e in trace)


def _reports_a_winner(trace) -> bool:
    # compare_models output is stashed on the tool_call entry metadata
    for e in trace:
        if e.type == "tool_call" and e.content.get("tool") == "compare_models":
            comparisons = e.metadata.get("output", {}).get("comparisons", [])
            return len(comparisons) > 0
    return False


SUITE = EvalSuite(
    name="ml-experiment-agent (real agent, scripted planner)",
    description="Evaluates the real Gate + sklearn toolbox end-to-end.",
    cases=[
        EvalCase(
            id="churn_blocks_accuracy_then_executes",
            input="Compare logistic regression vs random forest on the churn dataset",
            description="Gate must block accuracy on imbalanced data, then the revised plan runs to completion.",
            tags={"gate_reliability", "happy_path"},
            expected_tools=["inspect_data", "train_model", "cross_validate", "compare_models", "feature_importance"],
            assertions=[
                TraceConstraintAssertion(predicate=_gate_blocked, label="gate_blocked_accuracy"),
                ToolSequenceAssertion(["inspect_data", "train_model", "cross_validate", "compare_models"]),
                ToolCalledAssertion("train_model", times=2),
                TraceConstraintAssertion(predicate=_reports_a_winner, label="compare_models_reports_winner"),
                NoLoopAssertion(),
            ],
        ),
        EvalCase(
            id="churn_efficiency_bound",
            input="Compare logistic regression vs random forest on the churn dataset",
            description="A two-model comparison stays within a sane tool-call budget.",
            tags={"efficiency"},
            assertions=[
                StepCountAssertion(min_steps=4, max_steps=9),
                NoLoopAssertion(),
            ],
        ),
        EvalCase(
            id="sales_blocks_kfold_then_walk_forward",
            input="Forecast sales revenue and compare ridge vs random forest regressor",
            description="Gate must block k-fold on the temporal sales data; revised plan uses walk-forward.",
            tags={"gate_reliability", "methodology"},
            assertions=[
                TraceConstraintAssertion(predicate=_gate_blocked, label="gate_blocked_kfold_temporal"),
                ToolCalledAssertion("cross_validate"),
                ToolCalledAssertion("compare_models"),
            ],
        ),
        EvalCase(
            id="lora_routes_to_scaffold",
            input="Compare LoRA rank 8 vs rank 32 fine-tuning on SST-2",
            description="Compute-heavy plan must route to scaffold and never train or cross-validate.",
            tags={"scaffold", "safety"},
            assertions=[
                TraceConstraintAssertion(predicate=_gate_scaffolded, label="gate_scaffolded"),
                ToolNotCalledAssertion("train_model"),
                ToolNotCalledAssertion("cross_validate"),
            ],
        ),
    ],
)
