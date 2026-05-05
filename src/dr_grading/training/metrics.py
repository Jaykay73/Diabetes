"""Validation metrics for DR grading."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute QWK, the Kaggle APTOS competition metric."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def regression_to_classes(predictions: np.ndarray, thresholds: list[float] | None = None) -> np.ndarray:
    """Convert continuous grade predictions into integer DR classes."""

    values = np.asarray(predictions).reshape(-1)
    if thresholds is None:
        return np.rint(np.clip(values, 0.0, 4.0)).astype(int)
    if len(thresholds) != 4:
        raise ValueError("thresholds must contain four values")
    return np.digitize(values, bins=np.asarray(thresholds, dtype=float)).astype(int)


def logits_to_classes(logits: np.ndarray) -> np.ndarray:
    """Convert class logits/probabilities to predicted labels."""

    return np.asarray(logits).argmax(axis=1).astype(int)


def ordinal_logits_to_expected_grade(logits: np.ndarray) -> np.ndarray:
    """Convert ordered-threshold logits to expected continuous grades."""

    threshold_probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    return threshold_probs.sum(axis=1)
