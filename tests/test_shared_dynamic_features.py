from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import preprocessing.schema as schema
from preprocessing.feature_engineering import (
    STAGE_INDICATOR_COLUMN,
    frame_to_padded_stage_sequence,
    frame_to_stage_sequence,
)
from preprocessing.schema import DYNAMIC_FEATURES, TARGET_COLUMN, stage_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_DIMS = {1: 31, 2: 50, 3: 69, 4: 88}
ONLINE_SUFFIXES = [
    "hist_log_total_clicks",
    "hist_active_days",
    "hist_n_submitted",
    "hist_weighted_avg_score",
    "delta_log_total_clicks",
    "delta_active_days",
    "delta_avg_score",
    "trend_log_total_clicks",
    "trend_avg_score",
]


def test_dynamic_features_are_shared_by_every_stage_contract():
    for stage, expected in EXPECTED_STAGE_DIMS.items():
        columns = stage_feature_columns(stage)

        assert len(columns) == expected
        assert columns[-3:] == DYNAMIC_FEATURES
        assert all(columns.count(column) == 1 for column in DYNAMIC_FEATURES)
        assert not any(column.endswith(tuple(ONLINE_SUFFIXES)) for column in columns)


def test_baseline_views_exclude_dynamic_features_at_every_stage():
    assert hasattr(schema, "baseline_feature_columns")
    expected = {1: 28, 2: 47, 3: 66, 4: 85}

    for stage, size in expected.items():
        columns = schema.baseline_feature_columns(stage)

        assert len(columns) == size
        assert not set(DYNAMIC_FEATURES).intersection(columns)


def test_only_mamba_model_view_contains_dynamic_features():
    assert hasattr(schema, "mamba_feature_columns")
    assert hasattr(schema, "model_feature_columns")
    baseline_learners = ["lr", "svm", "knn", "rf", "xgb", "naive_bayes", "mlp", "tabnet"]

    for stage in range(1, 5):
        assert schema.model_feature_columns("mamba", stage) == schema.mamba_feature_columns(stage)
        assert set(DYNAMIC_FEATURES).issubset(schema.mamba_feature_columns(stage))
        for learner in baseline_learners:
            columns = schema.model_feature_columns(learner, stage)
            assert columns == schema.baseline_feature_columns(stage)
            assert not set(DYNAMIC_FEATURES).intersection(columns)


def test_mamba_sequence_repeats_dynamic_features_at_every_step():
    columns = stage_feature_columns(4)
    frame = pd.DataFrame(np.ones((2, len(columns))), columns=columns)
    frame.loc[:, DYNAMIC_FEATURES] = [2.0, 3.0, 4.0]

    sequence = frame_to_stage_sequence(frame, stage=4)
    staged = frame.copy()
    staged[STAGE_INDICATOR_COLUMN] = 4
    padded = frame_to_padded_stage_sequence(staged)

    assert sequence.shape == (2, 4, 31)
    assert padded.shape == (2, 4, 32)
    expected_dynamic = np.tile([2.0, 3.0, 4.0], (2, 4, 1))
    np.testing.assert_allclose(sequence[:, :, 9:12], expected_dynamic)


def test_stage_configs_declare_shared_dynamic_dimensions():
    for stage, expected in EXPECTED_STAGE_DIMS.items():
        config_path = PROJECT_ROOT / "configs" / f"stage{stage}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert config["expected_features"] == expected


def test_generated_stage_and_split_files_follow_shared_dynamic_contract():
    for stage, expected in EXPECTED_STAGE_DIMS.items():
        stage_name = f"stage{stage}"
        paths = [
            PROJECT_ROOT / "datasets" / stage_name / f"{stage_name}.csv",
            PROJECT_ROOT / "datasets" / "splits" / stage_name / "train.csv",
            PROJECT_ROOT / "datasets" / "splits" / stage_name / "test.csv",
        ]
        for path in paths:
            frame = pd.read_csv(path)

            assert frame.shape[1] == expected + 1
            assert frame.columns[-1] == TARGET_COLUMN
            assert all(column in frame.columns for column in DYNAMIC_FEATURES)
            assert not any(column.endswith(tuple(ONLINE_SUFFIXES)) for column in frame.columns)
