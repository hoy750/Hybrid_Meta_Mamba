from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class OptionalDependencyError(ImportError):
    """Raised when an optional model backend is not installed."""


def to_numpy(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def ensure_probability_matrix(probability: np.ndarray) -> np.ndarray:
    proba = np.asarray(probability, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] == 1:
        proba = np.column_stack([1.0 - proba[:, 0], proba[:, 0]])
    return proba


class ProbabilisticClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = ensure_probability_matrix(self.predict_proba(X))[:, 1]
        return (proba >= threshold).astype(int)


class SklearnBinaryClassifier(ProbabilisticClassifier):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return ensure_probability_matrix(self.estimator.predict_proba(X))
        if hasattr(self.estimator, "decision_function"):
            score = self.estimator.decision_function(X)
            prob = 1.0 / (1.0 + np.exp(-score))
            return ensure_probability_matrix(prob)
        pred = self.estimator.predict(X)
        return ensure_probability_matrix(pred)
