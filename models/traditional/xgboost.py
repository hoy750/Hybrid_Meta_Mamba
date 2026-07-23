from __future__ import annotations

from models.base import OptionalDependencyError, SklearnBinaryClassifier


class XGBoostClassifier(SklearnBinaryClassifier):
    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        **kwargs,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise OptionalDependencyError("Install xgboost to use XGBoostClassifier.") from exc
        estimator = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric="logloss",
            tree_method=kwargs.pop("tree_method", "hist"),
            verbosity=kwargs.pop("verbosity", 0),
            n_jobs=-1,
            **kwargs,
        )
        super().__init__(estimator)
