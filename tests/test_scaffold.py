import ast

import pytest

from agent.core.types import ExperimentRun, Plan
from agent.scaffold.generator import ScaffoldGenerator

scaffold = ScaffoldGenerator()


def _plan(methods, dataset="sst2", metric="accuracy", hyperparams=None):
    hp = hyperparams or {}
    return Plan(
        dataset=dataset,
        runs=[ExperimentRun(method=m, hyperparams=hp) for m in methods],
        metric=metric,
        cv_strategy="holdout",
        goal="test scaffold",
    )


# --- output structure --------------------------------------------------------

class TestOutputShape:
    def setup_method(self):
        self.result = scaffold(_plan(["lora"], hyperparams={"rank": 8}))

    def test_top_level_keys(self):
        assert {"dataset", "metric", "runs", "validation", "cost_estimate"} == self.result.keys()

    def test_one_run_entry(self):
        assert len(self.result["runs"]) == 1

    def test_run_keys(self):
        run = self.result["runs"][0]
        assert {"method", "config", "code"} == run.keys()

    def test_validation_keys(self):
        v = self.result["validation"]
        assert {"configs_valid", "issues"} == v.keys()

    def test_cost_estimate_mentions_not_run(self):
        assert "not run" in self.result["cost_estimate"].lower()


# --- lora scaffold -----------------------------------------------------------

class TestLoraScaffold:
    def setup_method(self):
        self.result = scaffold(_plan(["lora"], hyperparams={"rank": 8}))
        self.run = self.result["runs"][0]

    def test_config_rank_matches_hyperparams(self):
        assert self.run["config"]["rank"] == 8

    def test_alpha_defaults_to_twice_rank(self):
        assert self.run["config"]["alpha"] == 16

    def test_config_includes_dataset(self):
        assert self.run["config"]["dataset"] == "sst2"

    def test_code_is_valid_python(self):
        # the generated stub should at least parse without syntax errors
        try:
            ast.parse(self.run["code"])
        except SyntaxError as e:
            pytest.fail(f"generated code has syntax error: {e}")

    def test_code_contains_rank(self):
        assert "RANK" in self.run["code"]
        assert "8" in self.run["code"]

    def test_valid_config_passes_validation(self):
        assert self.result["validation"]["configs_valid"] is True
        assert self.result["validation"]["issues"] == []


# --- two-run scaffold (demo goal 2 shape) ------------------------------------

class TestTwoRunScaffold:
    def setup_method(self):
        plan = Plan(
            dataset="sst2",
            runs=[
                ExperimentRun("lora", {"rank": 8}),
                ExperimentRun("lora", {"rank": 32}),
            ],
            metric="accuracy",
            cv_strategy="holdout",
            goal="compare lora ranks",
        )
        self.result = scaffold(plan)

    def test_two_run_entries(self):
        assert len(self.result["runs"]) == 2

    def test_ranks_are_distinct(self):
        ranks = [r["config"]["rank"] for r in self.result["runs"]]
        assert ranks == [8, 32]

    def test_both_configs_valid(self):
        assert self.result["validation"]["configs_valid"] is True


# --- static validation catches bad configs -----------------------------------

class TestValidation:
    def test_invalid_rank_zero_raises_issue(self):
        result = scaffold(_plan(["lora"], hyperparams={"rank": 0}))
        assert result["validation"]["configs_valid"] is False
        assert any("rank" in issue for issue in result["validation"]["issues"])

    def test_rank_above_max_raises_issue(self):
        result = scaffold(_plan(["lora"], hyperparams={"rank": 1024}))
        assert result["validation"]["configs_valid"] is False

    def test_bad_learning_rate_raises_issue(self):
        result = scaffold(_plan(["lora"], hyperparams={"rank": 8, "learning_rate": 50.0}))
        assert result["validation"]["configs_valid"] is False
        assert any("learning_rate" in issue for issue in result["validation"]["issues"])

    def test_alpha_less_than_rank_warns(self):
        result = scaffold(_plan(["lora"], hyperparams={"rank": 32, "alpha": 4}))
        assert any("alpha" in issue for issue in result["validation"]["issues"])


# --- generic stub for unknown methods ----------------------------------------

class TestGenericStub:
    def test_unknown_method_produces_stub(self):
        result = scaffold(_plan(["my_custom_method"]))
        run = result["runs"][0]
        assert "TODO" in run["code"]

    def test_stub_is_valid_python(self):
        result = scaffold(_plan(["my_custom_method"]))
        ast.parse(result["runs"][0]["code"])
