from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    STAGE_LOCAL_SUFFIXES,
    TARGET_COLUMN,
    local_stage_features,
    model_feature_columns,
    stage_feature_columns,
    stage_output_name,
    validate_columns,
    validate_stage,
)

STAGE_INDICATOR_COLUMN = "__stage__"


def load_full_dataset(path: str | Path, target: str = TARGET_COLUMN) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} not found in {path}")
    validate_columns(frame.columns, stage=4, target=target)
    return frame


def build_stage_frame(
    frame: pd.DataFrame,
    stage: int,
    target: str = TARGET_COLUMN,
    include_target: bool = True,
) -> pd.DataFrame:
    validate_stage(stage)
    columns = stage_feature_columns(stage)
    validate_columns(frame.columns, stage=stage, target=target)
    selected = columns + ([target] if include_target else [])
    return frame.loc[:, selected].copy()


def write_stage_datasets(
    input_csv: str | Path,
    output_dir: str | Path,
    target: str = TARGET_COLUMN,
) -> dict[int, Path]:
    full = load_full_dataset(input_csv, target=target)
    output_root = Path(output_dir)
    written: dict[int, Path] = {}
    for stage in range(1, 5):
        stage_name = stage_output_name(stage)
        stage_dir = output_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        output_path = stage_dir / f"{stage_name}.csv"
        build_stage_frame(full, stage=stage, target=target).to_csv(output_path, index=False)
        written[stage] = output_path
    return written


def split_xy(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, np.ndarray]:
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} not found")
    return frame.drop(columns=[target]), frame[target].to_numpy(dtype=np.int64)


def select_model_features(
    frame: pd.DataFrame,
    model_name: str,
    stage: int,
) -> pd.DataFrame:
    columns = model_feature_columns(model_name, stage)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing model input columns for {model_name}: {missing[:10]}")
    return frame.loc[:, columns].copy()


def frame_to_stage_sequence(frame: pd.DataFrame, stage: int) -> np.ndarray:
    validate_stage(stage)
    validate_columns(frame.columns, stage=stage, require_target=False)
    common = frame.loc[:, STATIC_FEATURES + DYNAMIC_FEATURES].to_numpy(dtype=np.float32)
    frames = []
    for current_stage in range(1, stage + 1):
        local = frame.loc[:, local_stage_features(current_stage)].to_numpy(dtype=np.float32)
        frames.append(np.concatenate([common, local], axis=1))
    return np.stack(frames, axis=1)


def add_stage_indicator(frame: pd.DataFrame, stage: int) -> pd.DataFrame:
    validate_stage(stage)
    staged = frame.copy()
    staged[STAGE_INDICATOR_COLUMN] = stage
    return staged


def frame_to_padded_stage_sequence(
    frame: pd.DataFrame,
    stage: int | None = None,
    max_stage: int = 4,
) -> np.ndarray:
    validate_stage(max_stage)
    frame_dim = len(STATIC_FEATURES) + len(DYNAMIC_FEATURES) + len(STAGE_LOCAL_SUFFIXES)
    sequence = np.zeros((len(frame), max_stage, frame_dim + 1), dtype=np.float32)

    if STAGE_INDICATOR_COLUMN in frame.columns:
        stages = frame[STAGE_INDICATOR_COLUMN].to_numpy(dtype=np.int64)
    elif stage is not None:
        validate_stage(stage)
        stages = np.full(len(frame), stage, dtype=np.int64)
    else:
        raise ValueError(
            f"{STAGE_INDICATOR_COLUMN!r} column or explicit stage is required for padded stage sequence"
        )

    for current_stage in sorted(set(stages.tolist())):
        validate_stage(int(current_stage))
        if current_stage > max_stage:
            raise ValueError(f"stage {current_stage} exceeds max_stage {max_stage}")
        row_mask = stages == current_stage
        raw = frame_to_stage_sequence(frame.loc[row_mask], stage=int(current_stage))
        sequence[row_mask, : int(current_stage), :frame_dim] = raw
        sequence[row_mask, : int(current_stage), frame_dim] = 1.0
    return sequence


def numeric_feature_frame(frame: pd.DataFrame, stage: int) -> pd.DataFrame:
    columns = stage_feature_columns(stage)
    selected = frame.loc[:, columns].copy()
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected.isna().any().any():
        selected = selected.fillna(0.0)
    return selected
