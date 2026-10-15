import csv

import numpy as np
import pytest

from src.metrics import mse_by_time, rmse_by_time, summarise_mse
from src.results import RESULT_COLUMNS, append_result


def test_mse_rmse_and_summary_are_computed_per_time_step():
    truth = np.zeros((2, 2, 2), dtype="float32")
    rollout = np.array([np.ones((2, 2)) * 2, np.ones((2, 2)) * 4], dtype="float32")

    np.testing.assert_allclose(mse_by_time(truth, rollout), np.array([4.0, 16.0]))
    np.testing.assert_allclose(rmse_by_time(truth, rollout), np.array([2.0, 4.0]))
    assert summarise_mse(truth, rollout) == (10.0, 16.0)


def test_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mse_by_time(np.zeros((1, 2)), np.zeros((1, 3)))


def test_append_result_creates_expected_csv(tmp_path):
    csv_path = tmp_path / "results.csv"

    append_result(csv_path, "exp", 1.2, 3.4, 5, "hypothesis", "conclusion")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() == set(RESULT_COLUMNS)
    assert rows[0]["experiment_id"] == "exp"
    assert rows[0]["eval_mse_mean"] == "1.2"
    assert rows[0]["epochs_completed"] == "5"
