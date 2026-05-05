"""Threshold optimization for regression-style DR predictions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from dr_grading.training.metrics import quadratic_weighted_kappa, regression_to_classes


def optimize_thresholds(
    y_true: np.ndarray,
    predictions: np.ndarray,
    initial_thresholds: list[float] | None = None,
) -> list[float]:
    """Find thresholds that maximize validation QWK."""

    y_true = np.asarray(y_true).astype(int)
    predictions = np.asarray(predictions).reshape(-1)
    initial = np.asarray(initial_thresholds or [0.5, 1.5, 2.5, 3.5], dtype=float)

    def objective(thresholds: np.ndarray) -> float:
        sorted_thresholds = np.sort(thresholds)
        classes = regression_to_classes(predictions, sorted_thresholds.tolist())
        return -quadratic_weighted_kappa(y_true, classes)

    result = minimize(objective, initial, method="Nelder-Mead", options={"maxiter": 1000})
    return np.sort(result.x).clip(0.0, 4.0).tolist()
