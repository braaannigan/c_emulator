from pathlib import Path

import pytest

from src.config import ExperimentConfig, load_config

BASELINE = "configs/double_gyre_shifting_wind_2layer/20261015T120000-dgsw2l-baseline-state-cnn.yaml"


def test_load_baseline_config_exposes_core_options():
    config = load_config(BASELINE)

    assert config.experiment_id == "20261015T120000-dgsw2l-baseline-state-cnn"
    assert config.training_data == Path("data/double_gyre_shifting_wind_2layer/train/20260408T071500-period40d-duration5000d-2layer.nc")
    assert config.rollout_dir == Path("data/double_gyre_shifting_wind_2layer/rollout")
    assert config.experiment_log == Path("data/double_gyre_shifting_wind_2layer/logs/20261015T120000-dgsw2l-baseline-state-cnn.json")
    assert config.results_csv == Path("data/double_gyre_shifting_wind_2layer/results.csv")
    assert config.epochs == 20
    assert config.smoke_test_epochs == 1
    assert config.train_fraction == 0.8
    assert config.batch_size == 8
    assert config.learning_rate == 0.0001
    assert config.weight_decay == 0.0
    assert config.gradient_clip_norm is None
    assert config.lr_decay_milestones == []
    assert config.lr_decay_factor == 1.0
    assert config.rollout_steps is None
    assert config.eval_interval_epochs == 5
    assert config.save_rollout_outputs is False
    assert config.conclusion is None


def test_config_optional_values_are_cast_when_present():
    config = ExperimentConfig(
        raw={
            "experiment": {"id": "exp", "hypothesis": " h ", "conclusion": "done"},
            "paths": {"training_data": "train.nc", "rollout_dir": "rollout"},
            "results": {"csv_path": "results.csv"},
            "runtime": {},
            "split": {},
            "training": {"weight_decay": "0.01", "gradient_clip_norm": "1.5", "lr_decay_milestones": ["5", 10], "lr_decay_factor": "0.5"},
            "rollout": {"steps": "4", "eval_interval_epochs": "2", "save_outputs": True},
        },
        path=Path("config.yaml"),
    )

    assert config.results_csv == Path("results.csv")
    assert config.hypothesis == "h"
    assert config.conclusion == "done"
    assert config.seed == 42
    assert config.epochs == 60
    assert config.batch_size == 8
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.01
    assert config.gradient_clip_norm == 1.5
    assert config.lr_decay_milestones == [5, 10]
    assert config.lr_decay_factor == 0.5
    assert config.rollout_steps == 4
    assert config.eval_interval_epochs == 2
    assert config.save_rollout_outputs is True
    assert config.experiment_log == Path("logs/exp.json")


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)
