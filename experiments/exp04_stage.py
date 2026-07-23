from __future__ import annotations

import argparse

from common import load_config, resolve_path, save_metrics_table
from evaluation.statistics import collect_metric_files, summarize_by_stage


def run(config: dict) -> None:
    metrics = collect_metric_files(resolve_path(config["outputs"]["results_dir"]))
    summary = summarize_by_stage(metrics)
    save_metrics_table(summary.to_dict("records"), f"{config['outputs']['results_dir']}/tables/stage_wise_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: stage-wise dynamic analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
