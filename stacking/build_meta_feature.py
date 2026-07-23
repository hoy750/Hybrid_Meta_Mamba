from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing.schema import TARGET_COLUMN, legacy_model_name, meta_probability_columns


def legacy_probability_columns(learners: list[str], stage: int | None = None) -> list[str]:
    stages = [stage] if stage is not None else [1, 2, 3, 4]
    return [f"{legacy_model_name(learner)}_p_s{current_stage}" for learner in learners for current_stage in stages]


def available_probability_columns(meta_features: pd.DataFrame, learners: list[str], stage: int | None = None) -> list[str]:
    legacy_columns = legacy_probability_columns(learners, stage=stage)
    if set(legacy_columns).issubset(meta_features.columns):
        return legacy_columns
    modern_columns = meta_probability_columns(learners)
    if set(modern_columns).issubset(meta_features.columns):
        return modern_columns
    found = [column for column in legacy_columns + modern_columns if column in meta_features.columns]
    if found:
        return found
    raise ValueError(f"no probability columns found for learners: {learners}")


def build_meta_frame(
    probabilities: pd.DataFrame | np.ndarray,
    learners: list[str],
    y: np.ndarray | None = None,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    columns = meta_probability_columns(learners)
    if isinstance(probabilities, pd.DataFrame):
        frame = probabilities.loc[:, columns].copy() if set(columns).issubset(probabilities.columns) else probabilities.copy()
    else:
        frame = pd.DataFrame(probabilities, columns=columns)
    if y is not None:
        frame[target] = np.asarray(y, dtype=np.int64)
    return frame


def simple_average(meta_features: pd.DataFrame, learners: list[str]) -> np.ndarray:
    columns = available_probability_columns(meta_features, learners)
    return meta_features.loc[:, columns].mean(axis=1).to_numpy()


def weighted_average(meta_features: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    proba = np.zeros(len(meta_features), dtype=float)
    for learner, weight in weights.items():
        columns = available_probability_columns(meta_features, [learner])
        proba += meta_features.loc[:, columns].mean(axis=1).to_numpy() * (weight / total)
    return proba
