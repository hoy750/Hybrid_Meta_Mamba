from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from models.base import SklearnBinaryClassifier


class KNNClassifier(SklearnBinaryClassifier):
    """K-Nearest Neighbors — 论文中的"距离模型"基学习器。

    走 baseline 视图（不含 DYNAMIC_FEATURES）。类别不均衡场景下
    ``weights="distance"`` 比 ``"uniform"`` 更稳健；``n_jobs=-1`` 并行
    加速近邻搜索。
    """

    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        n_neighbors: int = 15,
        weights: str = "distance",
        metric: str = "minkowski",
        p: int = 2,
        **_,
    ):
        estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=-1,
        )
        super().__init__(estimator)
