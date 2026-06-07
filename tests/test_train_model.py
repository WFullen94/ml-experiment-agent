import pytest

from agent.tools.train_model import train_model


class TestChurnClassification:
    def setup_method(self):
        self.logreg = train_model("churn", "logreg")
        self.rf = train_model("churn", "random_forest")

    def test_returns_required_keys(self):
        for result in (self.logreg, self.rf):
            assert {"method", "dataset", "hyperparams", "n_train", "n_test", "scores"} <= result.keys()

    def test_classification_score_keys(self):
        for result in (self.logreg, self.rf):
            assert {"accuracy", "auc_pr", "f1"} == result["scores"].keys()

    def test_train_test_split_size(self):
        # 10k rows, 80/20 split
        for result in (self.logreg, self.rf):
            assert result["n_train"] == 8000
            assert result["n_test"] == 2000

    def test_dataset_label_correct(self):
        assert self.logreg["dataset"] == "churn"
        assert self.rf["dataset"] == "churn"

    def test_method_label_correct(self):
        assert self.logreg["method"] == "logreg"
        assert self.rf["method"] == "random_forest"

    def test_scores_are_between_zero_and_one(self):
        for result in (self.logreg, self.rf):
            for k, v in result["scores"].items():
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"

    def test_auc_pr_better_than_chance_on_imbalanced_data(self):
        # random classifier on 12:1 imbalance has AUC-PR ≈ 0.08
        # a real model should beat that comfortably
        assert self.logreg["scores"]["auc_pr"] > 0.15
        assert self.rf["scores"]["auc_pr"] > 0.15

    def test_hyperparams_merged_with_defaults(self):
        result = train_model("churn", "logreg", {"C": 0.1})
        assert result["hyperparams"]["C"] == 0.1
        assert "max_iter" in result["hyperparams"]   # default still present

    def test_custom_hyperparams_override_defaults(self):
        result = train_model("churn", "random_forest", {"n_estimators": 10})
        assert result["hyperparams"]["n_estimators"] == 10


class TestSalesRegression:
    def setup_method(self):
        self.result = train_model("sales", "rf_regressor")

    def test_regression_score_keys(self):
        assert {"mae", "rmse", "r2"} == self.result["scores"].keys()

    def test_r2_is_positive(self):
        # trend + seasonality signal should be learnable
        assert self.result["scores"]["r2"] > 0.0

    def test_temporal_split_preserves_row_order(self):
        # 5000 rows, last 20% held out
        assert self.result["n_train"] == 4000
        assert self.result["n_test"] == 1000


class TestErrors:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            train_model("churn", "xgboost_custom")

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            train_model("nonexistent", "logreg")
