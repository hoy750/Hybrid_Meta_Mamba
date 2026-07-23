from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import fit_evaluate_meta_model, load_config, load_train_test, resolve_path, save_metrics_table
from preprocessing.feature_engineering import split_xy
from preprocessing.schema import TARGET_COLUMN, stage_output_name
from stacking.generate_oof import generate_oof_predictions, predict_base_matrix


def run(config: dict) -> None:
    rows = []
    for stage in range(1, 5):
        train, test = load_train_test(config, stage)
        X_train, y_train = split_xy(train, target=config["data"].get("target", TARGET_COLUMN))
        X_test, y_test = split_xy(test, target=config["data"].get("target", TARGET_COLUMN))
        oof, fitted = generate_oof_predictions(
            X_train,
            y_train,
            learner_names=config["base_learners"],
            config=config,
            stage=stage,
            n_splits=config["training"].get("n_splits", 5),
        )
        meta_test = predict_base_matrix(fitted, X_test)
        meta_test[TARGET_COLUMN] = y_test
        stage_name = stage_output_name(stage)
        for meta_name in config["meta_learners"]:
            output_dir = f"{config['outputs']['results_dir']}/meta/{stage_name}/{meta_name}"
            metrics, _prediction, _model = fit_evaluate_meta_model(
                meta_name,
                config,
                stage,
                oof,
                meta_test,
                output_dir=output_dir,
            )
            rows.append(metrics)
        oof_dir = resolve_path(f"{config['outputs']['results_dir']}/oof/{stage_name}")
        Path(oof_dir).mkdir(parents=True, exist_ok=True)
        oof.to_csv(oof_dir / "oof.csv", index=False)
        pd.DataFrame(meta_test).to_csv(oof_dir / "test_meta.csv", index=False)
        save_metrics_table(rows, f"{config['outputs']['results_dir']}/meta/metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: meta learner comparison.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
