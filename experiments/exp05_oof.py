from __future__ import annotations

import argparse

import pandas as pd

from common import fit_evaluate_meta_model, load_config, load_train_test, save_metrics_table
from models.factory import create_model_from_config
from preprocessing.feature_engineering import select_model_features, split_xy
from preprocessing.schema import TARGET_COLUMN, stage_output_name
from stacking.generate_oof import generate_oof_predictions, predict_base_matrix


def leaky_meta_features(X_train, y_train, X_test, learners, config, stage):
    train_meta = pd.DataFrame(index=X_train.index)
    test_meta = pd.DataFrame(index=X_test.index)
    for learner in learners:
        train_view = select_model_features(X_train, learner, stage)
        test_view = select_model_features(X_test, learner, stage)
        model = create_model_from_config(learner, config=config, stage=stage)
        model.fit(train_view, y_train)
        train_meta[f"{learner}_prob"] = model.predict_proba(train_view)[:, 1]
        test_meta[f"{learner}_prob"] = model.predict_proba(test_view)[:, 1]
    return train_meta, test_meta


def run(config: dict) -> None:
    rows = []
    learners = config["base_learners"]
    for stage in range(1, 5):
        train, test = load_train_test(config, stage)
        X_train, y_train = split_xy(train, target=config["data"].get("target", TARGET_COLUMN))
        X_test, y_test = split_xy(test, target=config["data"].get("target", TARGET_COLUMN))

        oof_train, fitted = generate_oof_predictions(
            X_train,
            y_train,
            learner_names=learners,
            config=config,
            stage=stage,
            n_splits=config["training"].get("n_splits", 5),
        )
        oof_test = predict_base_matrix(fitted, X_test)
        oof_test[TARGET_COLUMN] = y_test

        leaky_train, leaky_test = leaky_meta_features(X_train, y_train, X_test, learners, config, stage)
        leaky_train[TARGET_COLUMN] = y_train
        leaky_test[TARGET_COLUMN] = y_test

        for label, meta_train, meta_test in [
            ("with_oof", oof_train, oof_test),
            ("without_oof", leaky_train, leaky_test),
        ]:
            stage_name = stage_output_name(stage)
            output_dir = f"{config['outputs']['results_dir']}/oof_verification/{label}/{stage_name}/meta_mamba"
            metrics, _prediction, _model = fit_evaluate_meta_model(
                "meta_mamba",
                config,
                stage,
                meta_train,
                meta_test,
                output_dir=output_dir,
            )
            metrics["strategy"] = label
            rows.append(metrics)
        save_metrics_table(rows, f"{config['outputs']['results_dir']}/oof_verification/metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5: OOF strategy verification.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
