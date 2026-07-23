import numpy as np

from evaluation.metrics import binary_metrics, tune_threshold


def test_binary_metrics_include_paper_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])

    metrics = binary_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["normal_recall"] == 1.0


def test_tune_threshold_returns_best_macro_f1_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.2, 0.6, 0.7, 0.8])

    threshold, metrics = tune_threshold(y_true, y_proba, step=0.1)

    assert 0.6 <= threshold <= 0.7
    assert metrics["macro_f1"] == 1.0
