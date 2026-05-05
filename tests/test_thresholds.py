import numpy as np

from dr_grading.inference.thresholds import optimize_thresholds


def test_optimize_thresholds_returns_four_sorted_values() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    preds = np.array([0.1, 0.2, 0.9, 1.2, 2.0, 2.1, 3.0, 3.1, 3.8, 4.0])

    thresholds = optimize_thresholds(y_true, preds)

    assert len(thresholds) == 4
    assert thresholds == sorted(thresholds)
