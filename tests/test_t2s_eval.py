import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t2s_eval import compute_metrics, max_wrong_direction, format_report_table


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


def test_format_report_table_includes_every_combo_and_metric():
    reports = {
        "mc_succ": {"success_scenario_summary": {"mae": 5.36, "rmse": 9.74}},
        "td0_succ": {"success_scenario_summary": {"mae": 4.72, "rmse": 7.34}},
    }
    table = format_report_table(reports, metrics=("mae", "rmse"))
    assert "mc_succ" in table and "td0_succ" in table
    assert "mae" in table and "rmse" in table
    assert "4.7200" in table  # value is actually rendered, not just the key


def test_format_report_table_handles_missing_metric_gracefully():
    reports = {"mc_succ": {"success_scenario_summary": {"mae": 5.36}}}
    table = format_report_table(reports, metrics=("mae", "spearman_rho"))
    assert "--" in table  # spearman_rho absent for this combo, shown as a placeholder, not a crash


def test_format_report_table_empty_reports_does_not_crash():
    table = format_report_table({}, metrics=("mae",))
    assert isinstance(table, str)


if __name__ == "__main__":
    test_perfect_predictions_give_zero_error_and_correlation_one()
    test_mae_matches_hand_computation()
    test_rmse_penalizes_large_errors_more_than_mae()
    test_max_wrong_direction_flags_countdown_going_up()
    test_max_wrong_direction_zero_for_monotonic_countdown()
    test_format_report_table_includes_every_combo_and_metric()
    test_format_report_table_handles_missing_metric_gracefully()
    test_format_report_table_empty_reports_does_not_crash()
    print("t2s_eval: all tests passed")
