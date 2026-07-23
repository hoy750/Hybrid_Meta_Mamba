from __future__ import annotations

from sklearn.naive_bayes import GaussianNB

from models.base import SklearnBinaryClassifier


class NaiveBayesClassifier(SklearnBinaryClassifier):
    """Gaussian Naive Bayes — 论文中的"概率模型"基学习器。

    输入是连续型 baseline 特征（含阶段的 local 统计量与静态字段），
    GaussianNB 对每个特征拟合独立高斯，先验类别用训练集频率。
    无需标准化、训练极快，作为强基线对照。
    """

    def __init__(
        self,
        stage: int | None = None,
        random_state: int = 42,
        var_smoothing: float = 1e-9,
        **_,
    ):
        estimator = GaussianNB(var_smoothing=var_smoothing)
        super().__init__(estimator)
