from __future__ import annotations

from collections.abc import Iterable


TARGET_COLUMN = "risk_2cls"

STATIC_FEATURES = [
    "disability_Y",
    "age_band_ord",
    "highest_education_ord",
    "imd_band_ord",
    "studied_credits",
    "num_of_prev_attempts",
    "days_to_register",
    "module_presentation_length",
    "cutoff_s4",
]

STAGE_LOCAL_SUFFIXES = [
    "active_days",
    "n_sites",
    "oucontent_clicks",
    "resource_clicks",
    "forumng_clicks",
    "homepage_clicks",
    "quiz_clicks",
    "subpage_clicks",
    "ouwiki_clicks",
    "url_clicks",
    "clicks_per_active",
    "longest_streak",
    "is_early_drop",
    "avg_score",
    "n_submitted",
    "max_score",
    "sqrt_active_days",
    "log_total_clicks",
    "click_entropy",
]

DYNAMIC_FEATURES = [
    "s_trend_clicks",
    "s_drop_flag",
    "s_total_clicks",
]

BASELINE_STAGE_DIMS = {1: 28, 2: 47, 3: 66, 4: 85}
MAMBA_STAGE_DIMS = {1: 31, 2: 50, 3: 69, 4: 88}
STAGE_DIMS = MAMBA_STAGE_DIMS
STAGE_NAMES = {1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4"}

# 9 baseline 选自论文表 1：覆盖 5 大类共 9 种代表模型。
BASE_LEARNERS = [
    "lr",           # 线性模型
    "svm",          # 核方法
    "knn",          # 距离模型
    "rf",           # Bagging
    "xgboost",      # Boosting
    "naive_bayes",  # 概率模型
    "mlp",          # 神经网络
    "tabnet",       # 表格深度学习
    "mamba",        # 状态空间模型
]

# 论文配置中仍可能使用旧短名，保留别名映射以便兼容历史 YAML / 旧脚本。
MODEL_ALIASES = {
    "xgb": "xgboost",
    "nb": "naive_bayes",
}

CANONICAL_TO_LEGACY_MODEL = {
    "xgboost": "xgb",
    "naive_bayes": "nb",
}

# Top-K 顺序按论文基线经验从强到弱：
# Boosting > SSM(Mamba) ≈ MLP > TabNet > Bagging > Kernel(SVM) > Linear > KNN > NB
LEGACY_TOPK_BASE_MODELS = {
    1: ["xgb"],
    2: ["xgb", "mamba"],
    3: ["xgb", "mamba", "mlp"],
    4: ["xgb", "mamba", "mlp", "tabnet"],
    5: ["xgb", "mamba", "mlp", "tabnet", "rf"],
    6: ["xgb", "mamba", "mlp", "tabnet", "rf", "svm"],
    7: ["xgb", "mamba", "mlp", "tabnet", "rf", "svm", "lr"],
    8: ["xgb", "mamba", "mlp", "tabnet", "rf", "svm", "lr", "knn"],
    9: ["xgb", "mamba", "mlp", "tabnet", "rf", "svm", "lr", "knn", "nb"],
}

# 9 个 meta learner 与 9 个 baseline 一一对应，并加入 MetaMambaClassifier
# 作为项目主打的元学习器。
META_LEARNERS = [
    "meta_mamba",
    "meta_lr",
    "meta_svm",
    "meta_knn",
    "meta_rf",
    "meta_naive_bayes",
    "meta_xgboost",
    "meta_mlp",
    "meta_tabnet",
]


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def legacy_model_name(name: str) -> str:
    return CANONICAL_TO_LEGACY_MODEL.get(name, name)


def validate_stage(stage: int) -> int:
    if stage not in STAGE_DIMS:
        raise ValueError(f"stage must be one of {sorted(STAGE_DIMS)}, got {stage!r}")
    return stage


def local_stage_features(stage: int) -> list[str]:
    validate_stage(stage)
    return [f"s{stage}_{suffix}" for suffix in STAGE_LOCAL_SUFFIXES]


def baseline_feature_columns(stage: int) -> list[str]:
    validate_stage(stage)
    columns = list(STATIC_FEATURES)
    for current_stage in range(1, stage + 1):
        columns.extend(local_stage_features(current_stage))
    expected = BASELINE_STAGE_DIMS[stage]
    if len(columns) != expected:
        raise AssertionError(
            f"baseline stage {stage} has {len(columns)} columns, expected {expected}"
        )
    return columns


def mamba_feature_columns(stage: int) -> list[str]:
    columns = baseline_feature_columns(stage) + list(DYNAMIC_FEATURES)
    expected = MAMBA_STAGE_DIMS[stage]
    if len(columns) != expected:
        raise AssertionError(
            f"mamba stage {stage} has {len(columns)} columns, expected {expected}"
        )
    return columns


def stage_feature_columns(stage: int) -> list[str]:
    return mamba_feature_columns(stage)


def model_feature_columns(model_name: str, stage: int) -> list[str]:
    canonical = canonical_model_name(model_name)
    if canonical not in BASE_LEARNERS:
        raise ValueError(f"model {model_name!r} is not a base learner")
    if canonical == "mamba":
        return mamba_feature_columns(stage)
    return baseline_feature_columns(stage)


def all_feature_columns() -> list[str]:
    return stage_feature_columns(4)


def stage_output_name(stage: int) -> str:
    validate_stage(stage)
    return STAGE_NAMES[stage]


def meta_probability_columns(learners: Iterable[str]) -> list[str]:
    return [f"{name}_prob" for name in learners]


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    available = set(columns)
    return [column for column in required if column not in available]


def validate_columns(
    columns: Iterable[str],
    stage: int,
    target: str = TARGET_COLUMN,
    require_target: bool = True,
) -> None:
    required = stage_feature_columns(stage)
    if require_target:
        required = required + [target]
    missing = missing_columns(columns, required)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"missing required columns for stage {stage}: {preview}{suffix}")


def infer_max_stage(columns: Iterable[str]) -> int:
    available = set(columns)
    for stage in (4, 3, 2, 1):
        if set(stage_feature_columns(stage)).issubset(available):
            return stage
    raise ValueError("could not infer a valid stage from provided columns")
