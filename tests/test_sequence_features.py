import numpy as np
import pandas as pd

from preprocessing.feature_engineering import (
    STAGE_INDICATOR_COLUMN,
    add_stage_indicator,
    frame_to_padded_stage_sequence,
    frame_to_stage_sequence,
)
from preprocessing.schema import TARGET_COLUMN, stage_feature_columns
from models.deep.mamba import SharedStageMambaClassifier


def test_frame_to_stage_sequence_repeats_static_features_per_stage():
    columns = stage_feature_columns(3) + [TARGET_COLUMN]
    frame = pd.DataFrame(np.ones((2, len(columns))), columns=columns)

    sequence = frame_to_stage_sequence(frame, stage=3)

    assert sequence.shape == (2, 3, 31)


def test_frame_to_stage_sequence_accepts_feature_only_frame():
    columns = stage_feature_columns(1)
    frame = pd.DataFrame(np.ones((2, len(columns))), columns=columns)

    sequence = frame_to_stage_sequence(frame, stage=1)

    assert sequence.shape == (2, 1, 31)


def test_frame_to_padded_stage_sequence_marks_visible_steps():
    stage1 = add_stage_indicator(
        pd.DataFrame(np.ones((1, len(stage_feature_columns(1)))), columns=stage_feature_columns(1)),
        stage=1,
    )
    stage3 = add_stage_indicator(
        pd.DataFrame(np.ones((1, len(stage_feature_columns(3)))), columns=stage_feature_columns(3)),
        stage=3,
    )
    frame = pd.concat([stage1, stage3], ignore_index=True)

    sequence = frame_to_padded_stage_sequence(frame)

    assert STAGE_INDICATOR_COLUMN in frame.columns
    assert sequence.shape == (2, 4, 32)
    assert sequence[0, :, -1].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert sequence[1, :, -1].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert np.all(sequence[0, 1:, :-1] == 0.0)


def test_shared_mamba_keeps_visibility_mask_unscaled():
    stage1 = add_stage_indicator(
        pd.DataFrame(np.ones((1, len(stage_feature_columns(1)))), columns=stage_feature_columns(1)),
        stage=1,
    )
    stage4 = add_stage_indicator(
        pd.DataFrame(np.ones((1, len(stage_feature_columns(4)))), columns=stage_feature_columns(4)),
        stage=4,
    )
    frame = pd.concat([stage1, stage4], ignore_index=True)
    model = SharedStageMambaClassifier(stage=None)

    features = model._prepare_features(frame, fit=True)

    assert features.shape == (2, 4, 32)
    assert features[0, :, -1].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features[1, :, -1].tolist() == [1.0, 1.0, 1.0, 1.0]
