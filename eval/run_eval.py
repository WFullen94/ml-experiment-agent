"""Run the agent-eval suite against the real ml-experiment-agent.

    python -m eval.run_eval

No ANTHROPIC_API_KEY needed — the LLM planner is replaced by ScriptedPlanner,
but the Gate, sklearn training/CV, and statistical model comparison are real.
Exits non-zero if any case fails, so it doubles as a CI gate.
"""
from __future__ import annotations

import sys

from agent_eval import EvalRunner, metrics

from .suite import SUITE, build_real_adapter


def main() -> int:
    adapter = build_real_adapter()
    report = EvalRunner(adapter=adapter).run(SUITE)
    report.print_summary()

    print("Trace-level metrics (computed over real trajectories):")
    metrics.summarize(report).print()

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
