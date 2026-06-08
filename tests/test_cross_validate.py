import pytest

from agent.tools.cross_validate import N_FOLDS, cross_validate


class TestKFoldClassification:
    def setup_method(self):
        self.result = cross_validate(
            dataset="churn",
            runs=["logreg", "random_forest"],
            strategy="kfold",
            metric="auc_pr",
        )

    def test_top_level_keys(self):
        assert {"results", "metric", "strategy", "n_folds"} == self.result.keys()

    def test_metric_and_strategy_labels(self):
        assert self.result["metric"] == "auc_pr"
        assert self.result["strategy"] == "kfold"
        assert self.result["n_folds"] == N_FOLDS

    def test_both_methods_in_results(self):
        assert "logreg" in self.result["results"]
        assert "random_forest" in self.result["results"]

    def test_each_method_has_required_keys(self):
        for method_result in self.result["results"].values():
            assert {"mean", "std", "fold_scores"} == method_result.keys()

    def test_fold_scores_length(self):
        for method_result in self.result["results"].values():
            assert len(method_result["fold_scores"]) == N_FOLDS

    def test_mean_consistent_with_fold_scores(self):
        import statistics
        for method_result in self.result["results"].values():
            computed = round(statistics.mean(method_result["fold_scores"]), 4)
            assert abs(computed - method_result["mean"]) < 0.001

    def test_scores_are_positive(self):
        # auc_pr is always positive
        for method_result in self.result["results"].values():
            assert method_result["mean"] > 0
            assert all(s >= 0 for s in method_result["fold_scores"])

    def test_rf_meaningfully_above_random_baseline(self):
        # random classifier on 12:1 imbalance → AUC-PR ≈ 0.08
        rf_mean = self.result["results"]["random_forest"]["mean"]
        assert rf_mean > 0.15


class TestWalkForwardRegression:
    def setup_method(self):
        self.result = cross_validate(
            dataset="sales",
            runs=["rf_regressor"],
            strategy="walk_forward",
            metric="mae",
        )

    def test_strategy_label(self):
        assert self.result["strategy"] == "walk_forward"

    def test_mae_is_positive(self):
        mae = self.result["results"]["rf_regressor"]["mean"]
        assert mae > 0

    def test_fold_scores_all_positive(self):
        # MAE is negated internally and flipped back — all values should be > 0
        scores = self.result["results"]["rf_regressor"]["fold_scores"]
        assert all(s > 0 for s in scores)


class TestSingleRun:
    def test_single_method_returns_single_result(self):
        result = cross_validate("churn", ["logreg"], "kfold", "f1")
        assert list(result["results"].keys()) == ["logreg"]


class TestErrors:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="unknown metric"):
            cross_validate("churn", ["logreg"], "kfold", "precision_at_k")

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown cv strategy"):
            cross_validate("churn", ["logreg"], "bootstrap", "accuracy")

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            cross_validate("nonexistent", ["logreg"], "kfold", "accuracy")
