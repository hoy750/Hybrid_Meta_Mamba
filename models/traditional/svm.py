from __future__ import annotations

from sklearn.svm import SVC

from models.base import SklearnBinaryClassifier


class SVMClassifier(SklearnBinaryClassifier):
    """Support Vector Machine (RBF kernel) — 论文中的"核方法"基学习器。

    输入特征遵循 baseline 视图（不含 DYNAMIC_FEATURES），通过 StandardScaler
    标准化在基学习器自身的 fit/predict 流程中由调用方保证（参见
    ``preprocessing.feature_engineering.select_model_features``）。
    SVM 对特征尺度敏感，因此保留 ``class_weight="balanced"`` 以应对类别不均衡。
    """

    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        class_weight: str | dict | None = "balanced",
        probability: bool = True,
        **_,
    ):
        estimator = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
        )
        super().__init__(estimator)
