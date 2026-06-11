from __future__ import annotations

import anthropic

from agent.core.types import ExperimentRun, GateOutcome, Plan, StepType, Trajectory

# Haiku for speed and cost in the planning loop; override with a stronger
# model if the plan quality is poor during week-1 testing.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = """\
You are an ML experiment planner. Your job is to decompose a research goal into a
structured experiment plan. Always call the submit_plan tool with your answer.

SUPPORTED DATASETS
  churn  — binary classification, 12:1 class imbalance, no time ordering
           → use auc_pr or f1 (NOT accuracy — imbalance makes it misleading)
  sales  — regression, daily time-series with trend and seasonality, time-ordered
           → use mae, rmse, or r2; use walk_forward CV (kfold leaks future data)
  sst2   — text classification, ~balanced, 67k sentences
           → use lora, qlora, or full_ft (scaffold-only; these won't be executed)

SUPPORTED METHODS
  Classification (fast): logreg, random_forest, gbm, small_mlp
  Regression    (fast):  ridge, rf_regressor, gbm_regressor
  LLM fine-tuning (scaffold-only): lora, qlora, full_ft

METHODOLOGY RULES (follow these to avoid gate rejections)
  1. Never use metric=accuracy on churn — use auc_pr or f1
  2. Never use cv_strategy=kfold on sales — use walk_forward
  3. Comparison goals need at least two runs
  4. lora/qlora/full_ft are always scaffold-only — that is expected behaviour

If a previous plan was rejected, the rejection reason appears in the conversation.
Revise the plan to address it exactly.\
"""

# Tool definition — forces the model to emit a valid Plan schema.
_SUBMIT_PLAN_TOOL = {
    "name": "submit_plan",
    "description": "Submit a structured ML experiment plan.",
    "input_schema": {
        "type": "object",
        "required": ["dataset", "runs", "metric", "cv_strategy"],
        "properties": {
            "dataset": {
                "type": "string",
                "enum": ["churn", "sales", "sst2"],
            },
            "runs": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["method"],
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": [
                                "logreg", "random_forest", "gbm", "small_mlp",
                                "ridge", "rf_regressor", "gbm_regressor",
                                "lora", "qlora", "full_ft",
                            ],
                        },
                        "hyperparams": {"type": "object"},
                    },
                },
            },
            "metric": {
                "type": "string",
                "enum": ["accuracy", "auc_pr", "f1", "mae", "rmse", "r2"],
            },
            "cv_strategy": {
                "type": "string",
                "enum": ["kfold", "walk_forward", "holdout"],
            },
        },
    },
}


class ApiPlanner:
    """
    Calls the Claude API to decompose a goal into a Plan.

    Replays prior plan attempts and gate rejections as a multi-turn
    conversation so the model can revise its proposal in context.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, goal: str, trajectory: Trajectory) -> Plan:
        messages = _build_messages(goal, trajectory)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            tools=[_SUBMIT_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "submit_plan"},
            messages=messages,
        )
        return _parse_response(response, goal)


# --- message construction ----------------------------------------------------

def _build_messages(goal: str, trajectory: Trajectory) -> list[dict]:
    """
    Build the messages list for the API call.

    On first call: a single user message with the goal.
    On revisions: replays previous (plan → gate rejection) turns so the
    model sees exactly why each plan was blocked.
    """
    messages: list[dict] = [{"role": "user", "content": goal}]

    plan_steps  = [s for s in trajectory if s.type == StepType.PLAN]
    gate_steps  = [s for s in trajectory if s.type == StepType.GATE_CHECK]

    for i, plan_step in enumerate(plan_steps):
        tool_use_id = f"toolu_plan_{i}"
        plan_dict   = plan_step.output.get("plan", {})

        # Previous plan as an assistant tool-use turn.
        messages.append({
            "role": "assistant",
            "content": [{
                "type":  "tool_use",
                "id":    tool_use_id,
                "name":  "submit_plan",
                "input": plan_dict,
            }],
        })

        # Corresponding gate rejection (if any) as a tool result.
        if i < len(gate_steps):
            gate = gate_steps[i]
            if gate.gate_result and gate.gate_result.outcome == GateOutcome.BLOCKED:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type":        "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": (
                            f"Gate rejected your plan: {gate.gate_result.reason}. "
                            f"Please revise and resubmit."
                        ),
                    }],
                })

    return messages


# --- response parsing --------------------------------------------------------

def _parse_response(response, goal: str) -> Plan:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_plan":
            data = block.input
            return Plan(
                dataset=data["dataset"],
                runs=[
                    ExperimentRun(
                        method=r["method"],
                        hyperparams=r.get("hyperparams") or {},
                    )
                    for r in data["runs"]
                ],
                metric=data["metric"],
                cv_strategy=data["cv_strategy"],
                goal=goal,
            )
    raise ValueError(
        "Planner API response did not include a submit_plan tool call. "
        "Check that tool_choice is set correctly."
    )
