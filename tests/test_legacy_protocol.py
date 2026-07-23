import numpy as np
import pandas as pd

from preprocessing.schema import (
    LEGACY_TOPK_BASE_MODELS,
    TARGET_COLUMN,
    canonical_model_name,
    stage_feature_columns,
)


def make_full_stage_frame(n_rows=40):
    rng = np.random.default_rng(42)
    columns = stage_feature_columns(4)
    frame = pd.DataFrame(rng.normal(size=(n_rows, len(columns))), columns=columns)
    frame[TARGET_COLUMN] = np.array([0, 1] * (n_rows // 2), dtype=np.int64)
    return frame


def test_schema_matches_legacy_stage_columns():
    assert stage_feature_columns(1)[:9] == [
        "disability_Y",
        "age_band_ord",
        "highest_education_ord",
        "imd_band_ord",
        "studied_credits",
        "num_of_prev_attempts",
        "days_to_register",
        "module_presentation_length",
        "cutoff_s4",
    ]
    assert [len(stage_feature_columns(stage)) for stage in range(1, 5)] == [31, 50, 69, 88]


def test_legacy_model_aliases_are_canonicalized():
    assert canonical_model_name("xgb") == "xgboost"
    assert canonical_model_name("nb") == "naive_bayes"
    assert canonical_model_name("naive_bayes") == "naive_bayes"


def test_legacy_topk_order_uses_old_short_names():
    assert LEGACY_TOPK_BASE_MODELS[2] == ["xgb", "mamba"]
    assert LEGACY_TOPK_BASE_MODELS[8] == [
        "xgb",
        "mamba",
        "mlp",
        "tabnet",
        "rf",
        "svm",
        "lr",
        "knn",
    ]


def test_factory_builds_legacy_aliases():
    from models.factory import create_model

    assert create_model("xgb").__class__.__name__ == "XGBoostClassifier"
    assert create_model("nb").__class__.__name__ == "NaiveBayesClassifier"


def test_legacy_tree_defaults_match_old_code():
    from models.factory import create_model

    xgb_model = create_model("xgb")
    svm_model = create_model("svm")
    knn_model = create_model("knn")

    assert xgb_model.estimator.get_params()["n_estimators"] == 200
    assert svm_model.estimator.get_params()["kernel"] == "rbf"
    assert svm_model.estimator.get_params()["probability"] is True
    assert knn_model.estimator.get_params()["n_neighbors"] == 15
    assert knn_model.estimator.get_params()["weights"] == "distance"


def test_legacy_oof_columns_are_four_stage():
    from stacking.generate_oof import legacy_oof_columns

    assert legacy_oof_columns(["xgb", "nb"]) == [
        "xgb_p_s1",
        "xgb_p_s2",
        "xgb_p_s3",
        "xgb_p_s4",
        "nb_p_s1",
        "nb_p_s2",
        "nb_p_s3",
        "nb_p_s4",
    ]


def test_legacy_split_helpers_match_expected_sizes():
    from preprocessing.split_train_test import (
        dataset_size_message,
        legacy_outer_split,
        legacy_train_val_test_split,
    )

    frame = make_full_stage_frame(n_rows=200)
    train, test = legacy_outer_split(frame, test_size=0.2, random_state=42)
    train_70, val_15, test_15 = legacy_train_val_test_split(frame, random_state=42)

    assert (len(train), len(test)) == (160, 40)
    assert (len(train_70), len(val_15), len(test_15)) == (140, 30, 30)
    assert "smoke-sized" in dataset_size_message(frame, full_size_hint=32593)


def test_generate_legacy_oof_four_stage_smoke():
    from stacking.generate_oof import generate_oof_predictions, predict_base_matrix

    frame = make_full_stage_frame(n_rows=40)
    X = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN].to_numpy()
    config = {
        "project": {"seed": 42},
        "model_params": {"lr": {"max_iter": 200, "class_weight": "balanced"}},
    }

    oof, models = generate_oof_predictions(
        X,
        y,
        learner_names=["lr"],
        config=config,
        n_splits=2,
    )
    test_matrix = predict_base_matrix(models, X.iloc[:5], learner_names=["lr"])

    assert list(oof.columns) == ["lr_p_s1", "lr_p_s2", "lr_p_s3", "lr_p_s4", TARGET_COLUMN]
    assert list(test_matrix.columns) == ["lr_p_s1", "lr_p_s2", "lr_p_s3", "lr_p_s4"]
    assert oof.shape == (40, 5)
    assert test_matrix.shape == (5, 4)


def test_meta_probability_mixin_restores_stage_model_axes():
    from models.deep.mamba import MetaMambaClassifier

    frame = pd.DataFrame(
        {
            "xgb_p_s1": [0.1, 0.2],
            "nb_p_s1": [0.3, 0.4],
            "xgb_p_s2": [0.5, 0.6],
            "nb_p_s2": [0.7, 0.8],
        }
    )

    sequence = MetaMambaClassifier()._raw_features(frame)

    assert sequence.shape == (2, 2, 2)
    np.testing.assert_allclose(sequence[0], [[0.1, 0.3], [0.5, 0.7]], atol=1e-6)
