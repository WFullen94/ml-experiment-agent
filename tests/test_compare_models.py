import pytest

from agent.tools.compare_models import compare_models


def _scores(mean_a, mean_b, n=5, spread=0.01):
    import numpy as np
    rng = np.random.default_rng(0)
    a = (rng.normal(0, spread, n) + mean_a).clip(0, 1).tolist()
    b = (rng.normal(0, spread, n) + mean_b).clip(0, 1).tolist()
    return a, b


class TestOutputShape:
    def setup_method(self):
        a, b = _scores(0.72, 0.85)
        self.result = compare_models({"logreg": a, "random_forest": b}, metric="auc_pr")

    def test_top_level_keys(self):
        assert {"metric", "comparisons", "overall_winner", "note"} == self.result.keys()

    def test_comparison_keys(self):
        c = self.result["comparisons"][0]
        expected = {
            "method_a", "method_b", "mean_a", "mean_b", "delta",
            "wilcoxon_p", "ttest_p", "significant", "tests_agree", "winner",
        }
        assert expected == c.keys()

    def test_one_comparison_for_two_methods(self):
        assert len(self.result["comparisons"]) == 1

    def test_metric_label_preserved(self):
        assert self.result["metric"] == "auc_pr"


class TestStatistics:
    def setup_method(self):
        a, b = _scores(0.72, 0.85)
        self.result = compare_models({"logreg": a, "random_forest": b}, metric="auc_pr")
        self.c = self.result["comparisons"][0]

    def test_delta_sign_matches_means(self):
        assert self.c["delta"] == round(self.c["mean_b"] - self.c["mean_a"], 4)

    def test_higher_mean_correctly_identified(self):
        assert self.c["mean_b"] > self.c["mean_a"]
        assert self.c["delta"] > 0

    def test_both_p_values_in_range(self):
        assert 0.0 <= self.c["wilcoxon_p"] <= 1.0
        assert 0.0 <= self.c["ttest_p"] <= 1.0

    def test_ttest_finds_significance_where_wilcoxon_cannot(self):
        # With n=5, Wilcoxon min p ≈ 0.0625 — can't reach 0.05.
        # Paired t-test has more power and should reach significance here.
        assert self.c["wilcoxon_p"] > 0.05
        assert self.c["ttest_p"] < 0.05

    def test_tests_agree_flag_reflects_disagreement(self):
        # Wilcoxon not significant, t-test is — they disagree
        assert self.c["tests_agree"] is False

    def test_significant_true_when_either_test_passes(self):
        assert self.c["significant"] is True

    def test_winner_is_rf(self):
        assert self.c["winner"] == "random_forest"
        assert self.result["overall_winner"] == "random_forest"

    def test_note_describes_both_tests(self):
        note = self.result["note"]
        assert "Wilcoxon" in note
        assert "t-test" in note


class TestIdenticalScores:
    def test_no_winner_when_identical(self):
        scores = [0.80, 0.81, 0.79, 0.80, 0.82]
        result = compare_models({"a": scores, "b": scores}, metric="auc_pr")
        c = result["comparisons"][0]
        assert c["wilcoxon_p"] == 1.0
        assert c["ttest_p"] == 1.0
        assert c["significant"] is False
        assert c["winner"] is None
        assert result["overall_winner"] is None


class TestEdgeCases:
    def test_three_methods_gives_three_comparisons(self):
        a, _ = _scores(0.70, 0.75)
        b, _ = _scores(0.80, 0.85)
        c, _ = _scores(0.85, 0.90)
        result = compare_models({"logreg": a, "random_forest": b, "gbm": c}, "auc_pr")
        assert len(result["comparisons"]) == 3

    def test_single_method_no_comparisons(self):
        result = compare_models({"logreg": [0.80, 0.81, 0.79, 0.80, 0.82]}, "auc_pr")
        assert result["comparisons"] == []
        assert result["overall_winner"] == "logreg"

    def test_tests_agree_when_both_significant(self):
        # Large difference over many "folds" — both tests should agree
        a = [0.60] * 10
        b = [0.90] * 10
        # all diffs identical → wilcoxon returns 1.0, ttest also 1.0
        # so use a slightly varied version
        a = [0.60 + i * 0.001 for i in range(10)]
        b = [0.90 + i * 0.001 for i in range(10)]
        result = compare_models({"a": a, "b": b}, "auc_pr")
        c = result["comparisons"][0]
        assert c["tests_agree"] is True
