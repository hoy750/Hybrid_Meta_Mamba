from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import binary_metrics, tune_threshold
from models.factory import create_model_from_config
from preprocessing.feature_engineering import (
    add_stage_indicator,
    build_stage_frame,
    load_full_dataset,
    select_model_features,
    split_xy,
)
from preprocessing.schema import BASE_LEARNERS, TARGET_COLUMN, canonical_model_name, stage_output_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config(path: str | Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_train_test(config: dict, stage: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_name = stage_output_name(stage)
    split_root = resolve_path(config["data"]["train_test_dir"])
    train_path = split_root / stage_name / "train.csv"
    test_path = split_root / stage_name / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    stage_path = resolve_path(config["data"]["datasets_dir"]) / stage_name / f"{stage_name}.csv"
    if stage_path.exists():
        frame = pd.read_csv(stage_path)
    else:
        full = load_full_dataset(resolve_path(config["data"]["raw_csv"]), target=config["data"].get("target", TARGET_COLUMN))
        frame = build_stage_frame(full, stage=stage, target=config["data"].get("target", TARGET_COLUMN))
    train, test = train_test_split(
        frame,
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["project"].get("seed", 42),
        stratify=frame[config["data"].get("target", TARGET_COLUMN)],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_evaluate_model(
    model_name: str,
    config: dict,
    stage: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame, object]:
    target = config["data"].get("target", TARGET_COLUMN)
    X_train_full, y_train_full = split_xy(train, target=target)
    X_test, y_test = split_xy(test, target=target)
    canonical = canonical_model_name(model_name)
    if canonical in BASE_LEARNERS:
        X_train_full = select_model_features(X_train_full, canonical, stage)
        X_test = select_model_features(X_test, canonical, stage)
    val_size = config["data"].get("validation_size", 0.2)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=config["project"].get("seed", 42),
        stratify=y_train_full,
    )
    model = create_model_from_config(model_name, config=config, stage=stage)
    model.fit(X_fit, y_fit)
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _val_metrics = tune_threshold(
        y_val,
        val_proba,
        step=config["training"].get("threshold_step", 0.01),
    )
    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = binary_metrics(y_test, test_proba, threshold=threshold)
    metrics.update({"stage": stage, "model": model_name})
    prediction = pd.DataFrame(
        {
            "sample_index": np.arange(len(test)),
            "ground_truth": y_test,
            "probability": test_proba,
            "prediction": (test_proba >= threshold).astype(int),
        }
    )
    if output_dir is not None:
        save_result(metrics, prediction, output_dir)
    return metrics, prediction, model


def fit_evaluate_shared_stage_model(
    model_name: str,
    config: dict,
    save: bool = True,
) -> dict[int, tuple[dict[str, float], pd.DataFrame, object]]:
    target = config["data"].get("target", TARGET_COLUMN)
    val_size = config["data"].get("validation_size", 0.2)
    seed = config["project"].get("seed", 42)
    fit_frames = []
    fit_labels = []
    validation_sets = {}
    test_sets = {}

    for stage in range(1, 5):
        train, test = load_train_test(config, stage)
        X_train_full, y_train_full = split_xy(train, target=target)
        X_test, y_test = split_xy(test, target=target)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=val_size,
            random_state=seed,
            stratify=y_train_full,
        )
        fit_frames.append(add_stage_indicator(X_fit, stage))
        fit_labels.append(y_fit)
        validation_sets[stage] = (add_stage_indicator(X_val, stage), y_val)
        test_sets[stage] = (add_stage_indicator(X_test, stage), y_test, len(test))

    model = create_model_from_config(model_name, config=config, stage=None)
    model.fit(pd.concat(fit_frames, ignore_index=True), np.concatenate(fit_labels))

    results = {}
    for stage in range(1, 5):
        X_val, y_val = validation_sets[stage]
        X_test, y_test, test_size = test_sets[stage]
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold, _val_metrics = tune_threshold(
            y_val,
            val_proba,
            step=config["training"].get("threshold_step", 0.01),
        )
        test_proba = model.predict_proba(X_test)[:, 1]
        metrics = binary_metrics(y_test, test_proba, threshold=threshold)
        metrics.update({"stage": stage, "model": model_name})
        prediction = pd.DataFrame(
            {
                "sample_index": np.arange(test_size),
                "ground_truth": y_test,
                "probability": test_proba,
                "prediction": (test_proba >= threshold).astype(int),
            }
        )
        if save:
            stage_name = stage_output_name(stage)
            save_result(
                metrics,
                prediction,
                f"{config['outputs']['results_dir']}/baseline/{stage_name}/{model_name}",
            )
        results[stage] = (metrics, prediction, model)
    return results


def fit_evaluate_meta_model(
    model_name: str,
    config: dict,
    stage: int,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, float], pd.DataFrame, object]:
    return fit_evaluate_model(model_name, config, stage, meta_train, meta_test, output_dir)


def save_result(metrics: dict[str, float], prediction: pd.DataFrame, output_dir: str | Path) -> None:
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    prediction.to_csv(output / "prediction.csv", index=False)


def save_metrics_table(rows: list[dict[str, float]], output_path: str | Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table
