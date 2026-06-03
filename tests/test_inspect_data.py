import pytest

from agent.tools.inspect_data import inspect_data


class TestChurn:
    def setup_method(self):
        self.result = inspect_data("churn")

    def test_shape_is_correct(self):
        n_rows, n_cols = self.result["shape"]
        assert n_rows == 10_000
        assert n_cols == 21   # 20 features + 1 target

    def test_n_features_excludes_target(self):
        assert self.result["n_features"] == 20

    def test_task_type(self):
        assert self.result["task_type"] == "classification"

    def test_no_time_index(self):
        assert self.result["has_time_index"] is False

    def test_imbalance_ratio_reflects_92_8_split(self):
        ratio = self.result["imbalance_ratio"]
        # make_classification with weights=[0.92, 0.08] → ~11–13:1
        assert 8.0 < ratio < 16.0

    def test_class_balance_sums_to_one(self):
        balance = self.result["class_balance"]
        assert abs(sum(balance.values()) - 1.0) < 0.01

    def test_missingness_all_zero(self):
        # synthetic data has no missing values
        assert all(v == 0.0 for v in self.result["missingness"].values())

    def test_dtypes_present_for_all_columns(self):
        n_rows, n_cols = self.result["shape"]
        assert len(self.result["dtypes"]) == n_cols


class TestSales:
    def setup_method(self):
        self.result = inspect_data("sales")

    def test_has_time_index(self):
        assert self.result["has_time_index"] is True

    def test_task_type(self):
        assert self.result["task_type"] == "regression"

    def test_imbalance_ratio_is_one_for_regression(self):
        assert self.result["imbalance_ratio"] == 1.0

    def test_class_balance_empty_for_regression(self):
        assert self.result["class_balance"] == {}

    def test_shape(self):
        n_rows, _ = self.result["shape"]
        assert n_rows == 5_000


class TestUnknownDataset:
    def test_raises_on_unregistered_dataset(self):
        with pytest.raises(KeyError, match="not registered"):
            inspect_data("nonexistent_dataset")
