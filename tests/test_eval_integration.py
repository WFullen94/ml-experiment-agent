"""Integration test: agent-eval evaluating the real ml-experiment-agent.

Runs the real AgentLoop (real Gate + real sklearn toolbox) behind the
deterministic ScriptedPlanner and asserts the agent-eval suite passes. No API
key required. Skipped automatically if agent-eval isn't installed.
"""
import pytest

pytest.importorskip("agent_eval", reason="agent-eval not installed; pip install the eval extra")

from agent_eval import EvalRunner, metrics  # noqa: E402

from eval.suite import SUITE, build_real_adapter  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return EvalRunner(adapter=build_real_adapter()).run(SUITE)


def test_all_cases_pass(report):
    assert report.failed == 0, [
        (r.case_id, [a.message for a in r.failed_assertions], r.error)
        for r in report.case_results if not r.passed
    ]


def test_gate_actually_revised(report):
    # The scripted planner proposes bad plans first; the real gate must block
    # them, forcing revisions. avg_revisions > 0 proves the loop really ran.
    assert metrics.avg_revisions(report) > 0


def test_tool_precision_recall_perfect(report):
    pr = metrics.tool_precision_recall(report)
    assert pr is not None
    precision, recall, _f1 = pr
    assert precision == 1.0
    assert recall == 1.0


def test_real_models_were_trained(report):
    # cross_validate ran real sklearn CV — its output carries fold scores.
    churn = next(r for r in report.case_results if r.case_id == "churn_blocks_accuracy_then_executes")
    cv_entries = [
        e for e in churn.trace
        if e.type == "tool_call" and e.content.get("tool") == "cross_validate"
    ]
    assert cv_entries
    results = cv_entries[0].metadata.get("output", {}).get("results", {})
    assert "logreg" in results and "random_forest" in results
    assert all("fold_scores" in v for v in results.values())
