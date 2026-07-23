from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metrics: dict[str, float]


def as_positive_probability(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]
    return proba


def binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    risk_proba = as_positive_probability(y_proba)
    y_pred = (risk_proba >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, risk_proba))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc": auc,
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "risk_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "threshold": float(threshold),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    step: float = 0.01,
    metric: str = "macro_f1",
) -> tuple[float, dict[str, float]]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_proba, threshold=best_threshold)
    thresholds = np.round(np.arange(0.0, 1.0 + step, step), 10)
    for threshold in thresholds:
        current = binary_metrics(y_true, y_proba, threshold=float(threshold))
        if current[metric] > best_metrics[metric]:
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold, best_metrics
