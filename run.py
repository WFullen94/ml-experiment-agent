"""
ml-experiment-agent  —  end-to-end demo
========================================
Runs two demo goals:

  Goal 1: churn comparison
    - Planner initially proposes accuracy → gate blocks it → planner revises
      to AUC-PR → executes → reports winner + feature importances

  Goal 2: LoRA rank comparison on SST-2
    - Compute budget check routes immediately to SCAFFOLD → configs generated
      with static validation

Environment
-----------
  ANTHROPIC_API_KEY must be set before running.

  export ANTHROPIC_API_KEY=sk-ant-...   # or add to .env
"""

from __future__ import annotations

import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# API key guard — clear message instead of a cryptic SDK error
# ---------------------------------------------------------------------------
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "\n[run.py] ANTHROPIC_API_KEY is not set.\n"
        "  export ANTHROPIC_API_KEY=sk-ant-<your-key>\n"
        "Set it in your shell and re-run.\n"
    )

from agent.core.loop import AgentLoop
from agent.core.planner import ApiPlanner
from agent.core.types import GateOutcome, StepType
from agent.gate.gate import Gate
from agent.scaffold.generator import ScaffoldGenerator
from agent.tools.compare_models import compare_models
from agent.tools.cross_validate import cross_validate
from agent.tools.feature_importance import feature_importance
from agent.tools.inspect_data import inspect_data
from agent.tools.train_model import train_model

# ---------------------------------------------------------------------------
# Toolbox
# ---------------------------------------------------------------------------
TOOLBOX = {
    "inspect_data":       inspect_data,
    "train_model":        train_model,
    "cross_validate":     cross_validate,
    "compare_models":     compare_models,
    "feature_importance": feature_importance,
}


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
_SEP = "─" * 72


def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_trajectory(trajectory) -> None:
    for step in trajectory:
        prefix = f"  [{step.step}] {step.type.value:<18}"

        if step.type == StepType.PLAN:
            plan = step.output.get("plan", {})
            runs  = [r["method"] for r in plan.get("runs", [])]
            print(f"{prefix}dataset={plan.get('dataset')}  "
                  f"runs={runs}  metric={plan.get('metric')}  "
                  f"cv={plan.get('cv_strategy')}")

        elif step.type == StepType.GATE_CHECK:
            gr = step.gate_result
            if gr:
                print(f"{prefix}outcome={gr.outcome.value}"
                      + (f"  reason={gr.reason!r}" if gr.reason else ""))

        elif step.type == StepType.TOOL_CALL:
            tool = step.input.get("tool", "?")
            if tool == "cross_validate":
                results = step.output.get("results", {})
                scores  = {m: f"{v.get('mean', 0):.4f}±{v.get('std', 0):.4f}"
                           for m, v in results.items()}
                print(f"{prefix}cross_validate  scores={scores}")
            elif tool == "compare_models":
                comparisons = step.output.get("comparisons", [])
                for c in comparisons:
                    sig = "SIGNIFICANT" if c.get("significant") else "not significant"
                    print(f"{prefix}compare  {c['model_a']} vs {c['model_b']}  "
                          f"p_wilcoxon={c.get('p_wilcoxon', 0):.3f}  "
                          f"p_ttest={c.get('p_ttest', 0):.3f}  [{sig}]")
            elif tool == "feature_importance":
                method  = step.input.get("args", {}).get("method", "?")
                top3    = list(step.output.get("importances", {}).items())[:3]
                print(f"{prefix}feature_importance  method={method}  "
                      f"top3={[f'{k}:{v:.3f}' for k, v in top3]}")
            else:
                print(f"{prefix}{tool}")

        elif step.type == StepType.SCAFFOLD:
            output  = step.output
            n_runs  = len(output.get("runs", []))
            valid   = output.get("validation", {}).get("configs_valid", "?")
            issues  = output.get("validation", {}).get("issues", [])
            cost    = output.get("cost_estimate", "")
            print(f"{prefix}scaffold  runs={n_runs}  valid={valid}  "
                  f"issues={issues}")
            print(f"  {'':18}cost: {cost}")
            for i, run in enumerate(output.get("runs", [])):
                cfg = run.get("config", {})
                print(f"  {'':18}run[{i}]: method={run['method']}  "
                      f"rank={cfg.get('rank')}  alpha={cfg.get('alpha')}  "
                      f"lr={cfg.get('learning_rate')}")

        elif step.type == StepType.RESULT:
            status = step.output.get("status", "?")
            print(f"{prefix}status={status}")
            if step.output.get("error"):
                print(f"  {'':18}error={step.output['error']!r}")


# ---------------------------------------------------------------------------
# Non-interactive confirm  (auto-approve for the demo)
# ---------------------------------------------------------------------------
def _auto_confirm(plan, estimated_cost: float) -> bool:
    print(f"\n  [confirm] auto-approving plan  "
          f"(cost estimate: ~{estimated_cost:.1f} GPU-min)")
    return True


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def _make_agent() -> AgentLoop:
    return AgentLoop(
        planner=ApiPlanner(),
        gate=Gate(),
        toolbox=TOOLBOX,
        scaffold_generator=ScaffoldGenerator(),
        confirm=_auto_confirm,
    )


# ---------------------------------------------------------------------------
# Demo goal 1 — churn classification comparison
# ---------------------------------------------------------------------------
def demo_goal_1() -> None:
    _header("Demo Goal 1 — Churn: logistic regression vs random forest")
    print(textwrap.dedent("""\
      Goal : Compare logistic regression vs random forest on the churn dataset.
      Expect: planner initially picks accuracy → gate blocks → revises to auc_pr/f1
              → executes → compare_models reports a winner.
    """))

    goal = (
        "Compare logistic regression vs random forest on the churn dataset "
        "and tell me which model performs better."
    )
    trajectory = _make_agent().run(goal)
    _print_trajectory(trajectory)


# ---------------------------------------------------------------------------
# Demo goal 2 — LoRA rank comparison (scaffold path)
# ---------------------------------------------------------------------------
def demo_goal_2() -> None:
    _header("Demo Goal 2 — SST-2: LoRA rank 8 vs rank 32")
    print(textwrap.dedent("""\
      Goal : Compare LoRA rank 8 vs rank 32 fine-tuning on SST-2 sentiment.
      Expect: compute_budget gate → SCAFFOLD outcome → configs generated with
              static validation, code stubs printed (nothing executed).
    """))

    goal = (
        "Compare LoRA rank 8 vs rank 32 fine-tuning on SST-2 sentiment "
        "and tell me which rank to use."
    )
    trajectory = _make_agent().run(goal)
    _print_trajectory(trajectory)

    # Print the generated code stubs so the demo is concrete
    scaffold_steps = [s for s in trajectory if s.type == StepType.SCAFFOLD]
    if scaffold_steps:
        runs = scaffold_steps[0].output.get("runs", [])
        for run in runs:
            print(f"\n  --- code stub: {run['method']} rank={run['config'].get('rank')} ---")
            for line in run["code"].splitlines()[:20]:
                print(f"  {line}")
            if len(run["code"].splitlines()) > 20:
                print("  ... (truncated)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_goal_1()
    demo_goal_2()
    print(f"\n{_SEP}")
    print("  Done.")
    print(_SEP)
