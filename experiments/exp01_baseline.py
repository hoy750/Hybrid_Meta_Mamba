from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import (
    fit_evaluate_model,
    fit_evaluate_shared_stage_model,
    load_config,
    load_train_test,
    save_metrics_table,
)
from preprocessing.schema import stage_output_name
from preprocessing.schema import BASE_LEARNERS, canonical_model_name
from preprocessing.split_train_test import dataset_size_message


SHARED_STAGE_LEARNERS = {"mamba"}


def select_base_learners(
    config: dict,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    learners = list(include) if include else list(config["base_learners"])
    excluded = set(exclude or [])
    selected = [name for name in learners if name not in excluded]
    if not selected:
        raise ValueError("no base learners selected")
    allowed = set(config["base_learners"]) | {name for name in BASE_LEARNERS} | {"xgb", "nb"}
    unknown = {name for name in selected if canonical_model_name(name) not in allowed and name not in allowed}
    if unknown:
        raise ValueError(f"unknown base learners: {sorted(unknown)}")
    return selected


def run(config: dict, learners: list[str] | None = None, save: bool = True) -> None:
    rows = []
    selected_learners = learners or config["base_learners"]
    shared_results = {}
    for model_name in selected_learners:
        if model_name in SHARED_STAGE_LEARNERS:
            print(f"training one shared {model_name} model across stage1-stage4")
            shared_results[model_name] = fit_evaluate_shared_stage_model(model_name, config, save=save)

    for stage in range(1, 5):
        train = None
        test = None
        stage_name = stage_output_name(stage)
        print(f"阶段名：{stage_name}")
        for model_name in selected_learners:
            if model_name in shared_results:
                metrics, _prediction, _model = shared_results[model_name][stage]
            else:
                if train is None or test is None:
                    train, test = load_train_test(config, stage)
                    if stage == 1 and train is not None:
                        print(dataset_size_message(train))
                output_dir = f"{config['outputs']['results_dir']}/baseline/{stage_name}/{model_name}"
                metrics, _prediction, _model = fit_evaluate_model(
                    model_name,
                    config,
                    stage,
                    train,
                    test,
                    output_dir=output_dir if save else None,
                )
            rows.append(metrics)
            print(
                f"stage{stage} {model_name}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"auc={metrics['auc']:.4f}, "
                f"normal_recall={metrics['normal_recall']:.4f}"
            )
        if save:
            save_metrics_table(rows, f"{config['outputs']['results_dir']}/baseline/metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: baseline comparison.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--include", nargs="*", default=None, help="Run only these base learners.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Skip these base learners, e.g. --exclude mamba.")
    parser.add_argument("--no-save", action="store_true", help="Run training/evaluation without writing result files.")
    args = parser.parse_args()
    config = load_config(args.config)
    learners = select_base_learners(config, include=args.include, exclude=args.exclude)
    run(config, learners=learners, save=not args.no_save)


if __name__ == "__main__":
    main()
