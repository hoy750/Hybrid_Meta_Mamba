import pandas as pd

from preprocessing.schema import TARGET_COLUMN, stage_feature_columns


def test_stage_feature_dimensions_match_paper_spec():
    assert [len(stage_feature_columns(stage)) for stage in range(1, 5)] == [31, 50, 69, 88]


def test_stage_columns_keep_target_separate():
    frame = pd.DataFrame(columns=stage_feature_columns(4) + [TARGET_COLUMN])

    assert TARGET_COLUMN not in stage_feature_columns(4)
    assert set(stage_feature_columns(4)).issubset(frame.columns)
