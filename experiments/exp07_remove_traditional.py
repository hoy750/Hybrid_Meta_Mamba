from __future__ import annotations

import argparse

from common import fit_evaluate_meta_model, load_config, load_train_test, save_metrics_table
from preprocessing.feature_engineering import split_xy
from preprocessing.schema import TARGET_COLUMN, stage_output_name
from stacking.generate_oof import generate_oof_predictions, predict_base_matrix


def run(config: dict) -> None:
    rows = []
    comparisons = {
        "only_mamba": ["mamba"],
        "hybrid_all": config["base_learners"],
    }
    for stage in range(1, 5):
        train, test = load_train_test(config, stage)
        X_train, y_train = split_xy(train, target=config["data"].get("target", TARGET_COLUMN))
        X_test, y_test = split_xy(test, target=config["data"].get("target", TARGET_COLUMN))
        for label, learners in comparisons.items():
            oof, fitted = generate_oof_predictions(
                X_train,
                y_train,
                learner_names=learners,
                config=config,
                stage=stage,
                n_splits=config["training"].get("n_splits", 5),
            )
            meta_test = predict_base_matrix(fitted, X_test)
            meta_test[TARGET_COLUMN] = y_test
            stage_name = stage_output_name(stage)
            output_dir = f"{config['outputs']['results_dir']}/remove_traditional/{label}/{stage_name}/meta_mamba"
            metrics, _prediction, _model = fit_evaluate_meta_model(
                "meta_mamba",
                config,
                stage,
                oof,
                meta_test,
                output_dir=output_dir,
            )
            metrics["strategy"] = label
            metrics["learners"] = ",".join(learners)
            rows.append(metrics)
        save_metrics_table(rows, f"{config['outputs']['results_dir']}/remove_traditional/metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7: remove traditional learners ablation.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
