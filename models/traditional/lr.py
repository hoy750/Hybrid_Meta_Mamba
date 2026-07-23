from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from models.base import SklearnBinaryClassifier


class LRClassifier(SklearnBinaryClassifier):
    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        **_,
    ):
        estimator = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        super().__init__(estimator)
