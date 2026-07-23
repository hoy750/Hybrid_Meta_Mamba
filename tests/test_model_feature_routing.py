import numpy as np
import pandas as pd

import experiments.common as experiment_common
import stacking.generate_oof as oof_module
from preprocessing.schema import DYNAMIC_FEATURES, TARGET_COLUMN, stage_feature_columns


class RecordingClassifier:
    fit_records: list[tuple[str, tuple[str, ...]]] = []
    predict_records: list[tuple[str, tuple[str, ...]]] = []

    def __init__(self, name: str):
        self.name = name

    def fit(self, X, y):
        self.fit_records.append((self.name, tuple(X.columns)))
        return self

    def predict_proba(self, X):
        self.predict_records.append((self.name, tuple(X.columns)))
        probability = np.full(len(X), 0.5, dtype=np.float64)
        return np.column_stack([1.0 - probability, probability])


def make_stage_frame(stage: int, rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    columns = stage_feature_columns(stage)
    frame = pd.DataFrame(rng.normal(size=(rows, len(columns))), columns=columns)
    frame[TARGET_COLUMN] = np.array([0, 1] * (rows // 2), dtype=np.int64)
    return frame


def reset_records():
    RecordingClassifier.fit_records = []
    RecordingClassifier.predict_records = []


def assert_model_columns_are_routed(records):
    dynamic = set(DYNAMIC_FEATURES)
    lr_columns = [set(columns) for name, columns in records if name == "lr"]
    mamba_columns = [set(columns) for name, columns in records if name == "mamba"]

    assert lr_columns
    assert mamba_columns
    assert all(not dynamic.intersection(columns) for columns in lr_columns)
    assert all(dynamic.issubset(columns) for columns in mamba_columns)


def test_experiment_common_routes_raw_features_by_model(monkeypatch):
    reset_records()
    frame = make_stage_frame(stage=1)
    train = frame.iloc[:30].reset_index(drop=True)
    test = frame.iloc[30:].reset_index(drop=True)
    config = {
        "project": {"seed": 42},
        "data": {"target": TARGET_COLUMN, "validation_size": 0.2},
        "training": {"threshold_step": 0.1},
    }
    monkeypatch.setattr(
        experiment_common,
        "create_model_from_config",
        lambda name, config, stage: RecordingClassifier(name),
    )

    experiment_common.fit_evaluate_model("lr", config, 1, train, test)
    experiment_common.fit_evaluate_model("mamba", config, 1, train, test)

    assert_model_columns_are_routed(RecordingClassifier.fit_records)
    assert_model_columns_are_routed(RecordingClassifier.predict_records)


def test_oof_training_and_prediction_route_features_by_learner(monkeypatch):
    reset_records()
    frame = make_stage_frame(stage=2)
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN].to_numpy()
    config = {"project": {"seed": 42}}
    monkeypatch.setattr(
        oof_module,
        "create_model_from_config",
        lambda name, config, stage: RecordingClassifier(name),
    )

    _oof, fitted = oof_module.generate_oof_predictions(
        X,
        y,
        learner_names=["lr", "mamba"],
        config=config,
        stage=2,
        n_splits=2,
    )
    oof_module.predict_base_matrix(
        fitted,
        X.iloc[:5],
        learner_names=["lr", "mamba"],
        stage=2,
    )

    assert_model_columns_are_routed(RecordingClassifier.fit_records)
    assert_model_columns_are_routed(RecordingClassifier.predict_records)
