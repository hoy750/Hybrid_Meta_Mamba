from __future__ import annotations

import importlib
from typing import Any

from preprocessing.schema import canonical_model_name


# 9 baseline 选自论文表 1 的 9 种代表模型：
#   线性模型  LR           — models.traditional.lr.LRClassifier
#   核方法    SVM          — models.traditional.svm.SVMClassifier
#   距离模型  KNN          — models.traditional.knn.KNNClassifier
#   Bagging  RF           — models.traditional.rf.RFClassifier
#   Boosting XGBoost      — models.traditional.xgboost.XGBoostClassifier
#   概率模型  Naive Bayes  — models.traditional.naive_bayes.NaiveBayesClassifier
#   神经网络  MLP          — models.deep.mlp.MLPClassifier
#   表格深度  TabNet       — models.deep.tabnet.TabNetModel
#   状态空间  Mamba        — models.deep.mamba.SharedStageMambaClassifier
MODEL_SPECS = {
    # —— 9 baseline ——
    "lr": ("models.traditional.lr", "LRClassifier"),
    "svm": ("models.traditional.svm", "SVMClassifier"),
    "knn": ("models.traditional.knn", "KNNClassifier"),
    "rf": ("models.traditional.rf", "RFClassifier"),
    "xgboost": ("models.traditional.xgboost", "XGBoostClassifier"),
    "naive_bayes": ("models.traditional.naive_bayes", "NaiveBayesClassifier"),
    "mlp": ("models.deep.mlp", "MLPClassifier"),
    "tabnet": ("models.deep.tabnet", "TabNetModel"),
    "mamba": ("models.deep.mamba", "SharedStageMambaClassifier"),
    # —— 7 meta learner（与 9 baseline 中非 Mamba 的 8 种基学习器族一一对应，
    #     并把主要重头 MetaMambaClassifier 放最前）——
    "meta_mamba": ("models.meta.meta_mamba", "MetaMambaClassifier"),
    "meta_lr": ("models.meta.meta_lr", "MetaLRClassifier"),
    "meta_svm": ("models.meta.meta_svm", "MetaSVMClassifier"),
    "meta_knn": ("models.meta.meta_knn", "MetaKNNClassifier"),
    "meta_rf": ("models.meta.meta_rf", "MetaRFClassifier"),
    "meta_naive_bayes": ("models.meta.meta_naive_bayes", "MetaNaiveBayesClassifier"),
    "meta_xgboost": ("models.meta.meta_xgboost", "MetaXGBoostClassifier"),
    "meta_mlp": ("models.meta.meta_mlp", "MetaMLPClassifier"),
    "meta_tabnet": ("models.meta.meta_tabnet", "MetaTabNetClassifier"),
}


def create_model(name: str, stage: int | None = None, random_state: int = 42, **params: Any):
    name = canonical_model_name(name)
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}. Available: {sorted(MODEL_SPECS)}")
    module_name, class_name = MODEL_SPECS[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(stage=stage, random_state=random_state, **params)


def create_model_from_config(
    name: str,
    config: dict,
    stage: int | None = None,
):
    canonical = canonical_model_name(name)
    params = dict(config.get("model_params", {}).get(canonical, {}))
    params.update(config.get("model_params", {}).get(name, {}))
    seed = config.get("project", {}).get("seed", 42)
    if canonical == "meta_mamba":
        params.update(config.get("model_params", {}).get("meta_mamba", {}))
    return create_model(canonical, stage=stage, random_state=seed, **params)
