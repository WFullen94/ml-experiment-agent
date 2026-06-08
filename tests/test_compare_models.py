import pytest

from agent.tools.compare_models import compare_models


def _scores(mean_a, mean_b, n=5, spread=0.01):
    """Generate plausible fold scores centred on given means."""
    import numpy as np
    rng = np.random.default_rng(0)
    a = (rng.normal(0, spread, n) + mean_a).clip(0, 1).tolist()
    b = (rng.normal(0, spread, n) + mean_b).clip(0, 1).tolist()
    return a, b


class TestTwoMethods:
    def setup_method(self):
        a, b = _scores(0.72, 0.85)
        self.result = compare_models({"logreg": a, "random_forest": b}, metric="auc_pr")

    def test_top_level_keys(self):
        assert {"metric", "comparisons", "overall_winner", "note"} == self.result.keys()

    def test_one_comparison_for_two_methods(self):
        assert len(self.result["comparisons"]) == 1

    def test_comparison_keys(self):
        c = self.result["comparisons"][0]
        assert {"method_a", "method_b", "mean_a", "mean_b", "delta",
                "p_value", "significant", "winner"} == c.keys()

    def test_delta_sign_matches_means(self):
        c = self.result["comparisons"][0]
        assert c["delta"] == round(c["mean_b"] - c["mean_a"], 4)

    def test_higher_mean_method_identified(self):
        # With 5 folds, Wilcoxon min achievable p ≈ 0.0625 — can't reach p<0.05.
        # The tool correctly reports the delta and means; significance requires more folds.
        c = self.result["comparisons"][0]
        assert c["mean_b"] > c["mean_a"]   # RF (b) > logreg (a)
        assert c["delta"] > 0
        assert 0.0 <= c["p_value"] <= 1.0

    def test_note_warns_about_limited_power(self):
        assert "note" in self.result
        assert "limited" in self.result["note"]

    def test_metric_label_preserved(self):
        assert self.result["metric"] == "auc_pr"


class TestIdenticalScores:
    def test_no_winner_when_identical(self):
        scores = [0.80, 0.81, 0.79, 0.80, 0.82]
        result = compare_models({"a": scores, "b": scores}, metric="auc_pr")
        c = result["comparisons"][0]
        assert c["p_value"] == 1.0
        assert c["significant"] is False
        assert c["winner"] is None
        assert result["overall_winner"] is None


class TestThreeMethods:
    def test_three_pairwise_comparisons(self):
        a, _ = _scores(0.70, 0.75)
        b, _ = _scores(0.80, 0.85)
        c, _ = _scores(0.85, 0.90)
        result = compare_models({"logreg": a, "random_forest": b, "gbm": c}, "auc_pr")
        assert len(result["comparisons"]) == 3


class TestSingleMethod:
    def test_single_method_no_comparisons(self):
        result = compare_models({"logreg": [0.80, 0.81, 0.79, 0.80, 0.82]}, "auc_pr")
        assert result["comparisons"] == []
        assert result["overall_winner"] == "logreg"
