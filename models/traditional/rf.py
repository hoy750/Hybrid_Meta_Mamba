from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from models.base import SklearnBinaryClassifier


class RFClassifier(SklearnBinaryClassifier):
    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int | None = 20,
        class_weight: str | dict | None = "balanced",
        **_,
    ):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        super().__init__(estimator)
