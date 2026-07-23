from experiments.exp01_baseline import select_base_learners
import experiments.exp01_baseline as exp01


def test_select_base_learners_can_exclude_mamba():
    config = {"base_learners": ["lr", "rf", "mamba"]}

    selected = select_base_learners(config, exclude=["mamba"])

    assert selected == ["lr", "rf"]


def test_baseline_reuses_one_shared_mamba_for_all_stages(monkeypatch):
    config = {
        "base_learners": ["lr", "mamba"],
        "outputs": {"results_dir": "results"},
    }
    calls = {"shared": 0, "regular": []}

    def fake_shared(model_name, config, save=True):
        calls["shared"] += 1
        assert model_name == "mamba"
        return {
            stage: (
                {
                    "stage": stage,
                    "model": "mamba",
                    "accuracy": 0.1,
                    "macro_f1": 0.2,
                    "auc": 0.3,
                    "normal_recall": 0.4,
                },
                None,
                object(),
            )
            for stage in range(1, 5)
        }

    def fake_fit(model_name, config, stage, train, test, output_dir=None):
        calls["regular"].append((model_name, stage))
        return (
            {
                "stage": stage,
                "model": model_name,
                "accuracy": 0.1,
                "macro_f1": 0.2,
                "auc": 0.3,
                "normal_recall": 0.4,
            },
            None,
            object(),
        )

    monkeypatch.setattr(exp01, "fit_evaluate_shared_stage_model", fake_shared)
    monkeypatch.setattr(exp01, "fit_evaluate_model", fake_fit)
    monkeypatch.setattr(exp01, "load_train_test", lambda config, stage: (None, None))
    monkeypatch.setattr(exp01, "save_metrics_table", lambda rows, output_path: None)

    exp01.run(config, learners=["lr", "mamba"], save=False)

    assert calls["shared"] == 1
    assert calls["regular"] == [("lr", 1), ("lr", 2), ("lr", 3), ("lr", 4)]
