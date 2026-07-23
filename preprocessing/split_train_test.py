from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.schema import TARGET_COLUMN, stage_output_name


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stratified_train_test_split(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def legacy_outer_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy 80/20 shuffle split used by the old OOF and Mamba scripts."""
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def legacy_train_val_test_split(
    frame: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy 70/15/15 shuffle split used for the historical baseline table."""
    if train_size <= 0 or val_size <= 0 or train_size + val_size >= 1:
        raise ValueError("train_size and val_size must be positive and leave room for test")
    train, temp = train_test_split(
        frame,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )
    val_count = int(round(len(frame) * val_size))
    val, test = train_test_split(
        temp,
        train_size=val_count,
        random_state=random_state,
        shuffle=True,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def dataset_size_message(frame: pd.DataFrame, full_size_hint: int = 32593) -> str:
    rows = len(frame)
    if rows < 1000:
        return (
            f"dataset is smoke-sized ({rows} rows); restore the full "
            f"{full_size_hint:,}-row CSV before comparing paper metrics"
        )
    if rows != full_size_hint:
        return f"dataset has {rows:,} rows; expected paper-scale hint is {full_size_hint:,}"
    return f"dataset has the expected full size ({rows:,} rows)"


def split_stage_files(
    datasets_dir: str | Path,
    output_dir: str | Path,
    target: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[int, tuple[Path, Path]]:
    datasets_root = Path(datasets_dir)
    output_root = Path(output_dir)
    written: dict[int, tuple[Path, Path]] = {}
    for stage in range(1, 5):
        stage_name = stage_output_name(stage)
        input_path = datasets_root / stage_name / f"{stage_name}.csv"
        frame = pd.read_csv(input_path)
        train, test = stratified_train_test_split(
            frame,
            target=target,
            test_size=test_size,
            random_state=random_state,
        )
        stage_dir = output_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        train_path = stage_dir / "train.csv"
        test_path = stage_dir / "test.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        written[stage] = (train_path, test_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/test splits for all stages.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    written = split_stage_files(
        datasets_dir=config["data"]["datasets_dir"],
        output_dir=config["data"]["train_test_dir"],
        target=config["data"].get("target", TARGET_COLUMN),
        test_size=config["data"].get("test_size", 0.2),
        random_state=config["project"].get("seed", 42),
    )
    for stage, (train_path, test_path) in written.items():
        print(f"stage{stage}: train={train_path} test={test_path}")


if __name__ == "__main__":
    main()
