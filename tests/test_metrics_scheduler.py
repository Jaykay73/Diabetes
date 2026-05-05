import numpy as np

from dr_grading.training.metrics import quadratic_weighted_kappa, regression_to_classes


def test_qwk_perfect_predictions() -> None:
    labels = np.array([0, 1, 2, 3, 4])

    score = quadratic_weighted_kappa(labels, labels)

    assert score == 1.0


def test_regression_to_classes_with_thresholds() -> None:
    predictions = np.array([0.1, 0.8, 1.8, 2.9, 3.7])

    classes = regression_to_classes(predictions, thresholds=[0.5, 1.5, 2.5, 3.5])

    assert classes.tolist() == [0, 1, 2, 3, 4]
