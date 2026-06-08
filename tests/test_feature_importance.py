import pytest

from agent.tools.feature_importance import TOP_N, feature_importance


class TestLogReg:
    def setup_method(self):
        self.result = feature_importance("churn", "logreg")

    def test_top_level_keys(self):
        assert {"method", "dataset", "importances", "n_features", "importance_type"} == self.result.keys()

    def test_method_and_dataset_labels(self):
        assert self.result["method"] == "logreg"
        assert self.result["dataset"] == "churn"

    def test_native_importance_type(self):
        assert self.result["importance_type"] == "native"

    def test_returns_top_n_features(self):
        assert len(self.result["importances"]) == TOP_N

    def test_importance_entry_keys(self):
        for entry in self.result["importances"]:
            assert {"feature", "importance"} == entry.keys()

    def test_importances_are_non_negative(self):
        for entry in self.result["importances"]:
            assert entry["importance"] >= 0

    def test_sorted_descending(self):
        scores = [e["importance"] for e in self.result["importances"]]
        assert scores == sorted(scores, reverse=True)

    def test_n_features_correct(self):
        assert self.result["n_features"] == 20


class TestRandomForest:
    def setup_method(self):
        self.result = feature_importance("churn", "random_forest")

    def test_native_importance_type(self):
        assert self.result["importance_type"] == "native"

    def test_importances_sum_to_roughly_one(self):
        # Tree importances sum to 1.0 — top-10 won't sum to exactly 1 but should be close
        total = sum(e["importance"] for e in self.result["importances"])
        assert total > 0.5


class TestUnknownInputs:
    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            feature_importance("nonexistent", "logreg")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            feature_importance("churn", "xgboost")
