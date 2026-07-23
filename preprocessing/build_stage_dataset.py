from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.feature_engineering import write_stage_datasets
from preprocessing.schema import TARGET_COLUMN


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage1-stage4 datasets from the 88-feature CSV.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_csv = args.input_csv or config["data"]["raw_csv"]
    output_dir = args.output_dir or config["data"]["datasets_dir"]
    target = config.get("data", {}).get("target", TARGET_COLUMN)

    written = write_stage_datasets(input_csv=input_csv, output_dir=output_dir, target=target)
    for stage, path in written.items():
        print(f"stage{stage}: {path}")


if __name__ == "__main__":
    main()
