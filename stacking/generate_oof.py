from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.factory import create_model_from_config
from preprocessing.feature_engineering import select_model_features, split_xy
from preprocessing.schema import (
    LEGACY_TOPK_BASE_MODELS,
    TARGET_COLUMN,
    infer_max_stage,
    legacy_model_name,
    stage_feature_columns,
    stage_output_name,
)
from preprocessing.split_train_test import dataset_size_message


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config(path: str | Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def legacy_oof_columns(learner_names: list[str], max_stage: int = 4) -> list[str]:
    columns: list[str] = []
    for learner in learner_names:
        legacy = legacy_model_name(learner)
        for stage in range(1, max_stage + 1):
            columns.append(f"{legacy}_p_s{stage}")
    return columns


def _stage_frame(X: pd.DataFrame, stage: int, learner_name: str) -> pd.DataFrame:
    return select_model_features(X, learner_name, stage)


def generate_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    learner_names: list[str],
    config: dict,
    stage: int | None = None,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, dict[int, object]]]:
    """Generate legacy-style OOF columns for every visible stage.

    The old project trained a separate model per stage and emitted columns such
    as `xgb_p_s1` ... `xgb_p_s4`. This function keeps that protocol while
    using the new project's model factory.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    max_stage = stage or infer_max_stage(X.columns)
    seed = config.get("project", {}).get("seed", 42)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = np.asarray(y, dtype=np.int64)
    oof_data: dict[str, np.ndarray] = {}
    fitted_models: dict[str, dict[int, object]] = {}

    for learner in learner_names:
        legacy = legacy_model_name(learner)
        fitted_models[legacy] = {}
        for current_stage in range(1, max_stage + 1):
            column = f"{legacy}_p_s{current_stage}"
            fold_probability = np.zeros(len(X), dtype=np.float32)
            X_stage = _stage_frame(X, current_stage, learner)
            for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(X_stage), start=1):
                model = create_model_from_config(learner, config=config, stage=current_stage)
                model.fit(X_stage.iloc[train_idx], y[train_idx])
                fold_probability[valid_idx] = model.predict_proba(X_stage.iloc[valid_idx])[:, 1]
                print(f"stage{current_stage} {legacy} fold {fold_id}/{n_splits} done")
            oof_data[column] = fold_probability

            full_model = create_model_from_config(learner, config=config, stage=current_stage)
            full_model.fit(X_stage, y)
            fitted_models[legacy][current_stage] = full_model

    oof = pd.DataFrame(oof_data, index=X.index)
    oof[TARGET_COLUMN] = y
    return oof.loc[:, legacy_oof_columns(learner_names, max_stage=max_stage) + [TARGET_COLUMN]], fitted_models


def predict_base_matrix(
    fitted_models: dict[str, dict[int, object]],
    X: pd.DataFrame,
    learner_names: list[str] | None = None,
    stage: int | None = None,
) -> pd.DataFrame:
    if learner_names is None:
        learner_names = list(fitted_models)
    max_stage = stage or max(max(per_stage) for per_stage in fitted_models.values())
    data: dict[str, np.ndarray] = {}
    for learner in learner_names:
        legacy = legacy_model_name(learner)
        per_stage = fitted_models[legacy]
        for current_stage in range(1, max_stage + 1):
            model = per_stage[current_stage]
            X_stage = _stage_frame(X, current_stage, learner)
            data[f"{legacy}_p_s{current_stage}"] = model.predict_proba(X_stage)[:, 1]
    return pd.DataFrame(data, index=X.index)


def select_learners(config: dict, include: list[str] | None, exclude: list[str] | None, topk: int | None) -> list[str]:
    if topk is not None:
        if topk not in LEGACY_TOPK_BASE_MODELS:
            raise ValueError(f"topk must be one of {sorted(LEGACY_TOPK_BASE_MODELS)}")
        learners = list(LEGACY_TOPK_BASE_MODELS[topk])
    elif include:
        learners = list(include)
    else:
        learners = list(config["base_learners"])
    excluded = set(exclude or [])
    learners = [learner for learner in learners if learner not in excluded]
    if not learners:
        raise ValueError("no learners selected")
    return learners


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate legacy four-stage K-Fold OOF meta features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", type=int, default=4, help="Maximum visible stage. Use 4 for legacy stacking.")
    parser.add_argument("--input", default=None, help="Optional CSV with full stage columns.")
    parser.add_argument("--output", default=None, help="Optional OOF output CSV.")
    parser.add_argument("--include", nargs="*", default=None, help="Run only these base learners.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Skip these base learners.")
    parser.add_argument("--topk", type=int, default=None, help="Use legacy Top-K base learner order.")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    stage_name = stage_output_name(args.stage)
    input_path = (
        resolve_path(args.input)
        if args.input
        else resolve_path(config["data"]["train_test_dir"]) / stage_name / "train.csv"
    )
    output_path = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config["outputs"]["results_dir"]) / "oof" / stage_name / "oof.csv"
    )

    frame = pd.read_csv(input_path)
    print(dataset_size_message(frame))
    X, y = split_xy(frame, target=config["data"].get("target", TARGET_COLUMN))
    learners = select_learners(config, include=args.include, exclude=args.exclude, topk=args.topk)
    oof, _models = generate_oof_predictions(
        X,
        y,
        learner_names=learners,
        config=config,
        stage=args.stage,
        n_splits=args.n_splits or config["training"].get("n_splits", 5),
    )
    print(f"oof shape={oof.shape} columns={list(oof.columns)}")
    if not args.no_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(output_path, index=False)
        print(output_path)


if __name__ == "__main__":
    main()
