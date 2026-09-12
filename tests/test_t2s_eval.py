import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t2s_eval import compute_metrics, max_wrong_direction


def test_perfect_predictions_give_zero_error_and_correlation_one():
    truths = np.array([10, 8, 6, 4, 2, 0], dtype=np.float32)
    m = compute_metrics(truths.copy(), truths)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert abs(m["pearson_r"] - 1.0) < 1e-9
    assert abs(m["spearman_rho"] - 1.0) < 1e-9


def test_mae_matches_hand_computation():
    preds = np.array([1.0, 2.0, 3.0])
    truths = np.array([0.0, 0.0, 0.0])
    m = compute_metrics(preds, truths)
    assert abs(m["mae"] - 2.0) < 1e-9   # mean(|1|,|2|,|3|) = 2


def test_rmse_penalizes_large_errors_more_than_mae():
    preds = np.array([1.0, 1.0, 1.0, 10.0])
    truths = np.array([0.0, 0.0, 0.0, 0.0])
    m = compute_metrics(preds, truths)
    assert m["rmse"] > m["mae"]


def test_max_wrong_direction_flags_countdown_going_up():
    curve = np.array([10, 8, 6, 9, 4])  # jumps up at index 3
    assert max_wrong_direction(curve) == 3.0  # 9 - 6


def test_max_wrong_direction_zero_for_monotonic_countdown():
    curve = np.array([10, 8, 6, 4, 2, 0])
    assert max_wrong_direction(curve) <= 0.0


if __name__ == "__main__":
    test_perfect_predictions_give_zero_error_and_correlation_one()
    test_mae_matches_hand_computation()
    test_rmse_penalizes_large_errors_more_than_mae()
    test_max_wrong_direction_flags_countdown_going_up()
    test_max_wrong_direction_zero_for_monotonic_countdown()
    print("t2s_eval: all tests passed")
