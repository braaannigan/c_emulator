import json

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from src.config import load_config
from src.data_state import fit_channel_normalisation
from src.models.baseline_cnn import StateCNN
from src.pipelines.dgsw2l_baseline import (
    StateCNNEmulator,
    build_state_model,
    current_learning_rate,
    decay_learning_rate,
    resolve_device,
    run_experiment,
)


def tiny_two_layer_dataset(n_times=12):
    rng = np.random.default_rng(0)
    return xr.Dataset(
        data_vars={
            "layer_thickness": (("time_days", "layers", "y", "x"), (100 + rng.normal(size=(n_times, 2, 4, 4))).astype("float32")),
            "zonal_velocity": (("time_days", "layers", "y", "xp1"), rng.normal(size=(n_times, 2, 4, 5)).astype("float32")),
            "meridional_velocity": (("time_days", "layers", "yp1", "x"), rng.normal(size=(n_times, 2, 5, 4)).astype("float32")),
        },
        coords={
            "time_days": np.arange(n_times, dtype="float64") * 7,
            "layers": [0, 1],
            "y": np.arange(4, dtype="float64"),
            "x": np.arange(4, dtype="float64"),
            "xp1": np.arange(5),
            "yp1": np.arange(5),
        },
        attrs={"wind_stress_max": 0.05, "wind_shift_amplitude_m": 1.0, "wind_shift_period_days": 40.0},
    )


def write_config(tmp_path, **overrides):
    data_path = tmp_path / "train.nc"
    tiny_two_layer_dataset().to_netcdf(data_path)
    raw = {
        "experiment": {"id": "tiny", "acronym": "dgsw2l", "family": "double_gyre_shifting_wind_2layer", "hypothesis": "h", "conclusion": None},
        "paths": {
            "training_data": str(data_path),
            "rollout_dir": str(tmp_path / "rollout"),
            "experiment_log": str(tmp_path / "logs" / "tiny.json"),
            "results_csv": str(tmp_path / "results.csv"),
        },
        "runtime": {"seed": 1, "device": "cpu", "epochs": 2, "smoke_test_epochs": 1},
        "split": {"train_fraction": 0.75},
        "training": {"batch_size": 4, "learning_rate": 0.001, "gradient_clip_norm": 1.0, "lr_decay_milestones": [1], "lr_decay_factor": 0.5},
        "model": {"hidden_channels": 4, "hidden_layers": 1},
        "rollout": {"eval_interval_epochs": 1, **overrides},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_state_cnn_validates_arguments_and_shapes():
    model = StateCNN(7, 6, hidden_channels=4, hidden_layers=2)
    assert model(torch.zeros(1, 7, 4, 4)).shape == (1, 6, 4, 4)
    with pytest.raises(ValueError):
        StateCNN(7, 6, hidden_layers=0)
    with pytest.raises(ValueError):
        StateCNN(7, 6, kernel_size=4)


def test_build_state_model_uses_config_values():
    model = build_state_model({"hidden_channels": 5, "hidden_layers": 3, "kernel_size": 5}, 7, 6)
    convs = [layer for layer in model.net if isinstance(layer, torch.nn.Conv2d)]
    assert len(convs) == 4
    assert convs[0].out_channels == 5
    assert convs[0].kernel_size == (5, 5)


def test_learning_rate_helpers_and_device():
    optimiser = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    decay_learning_rate(optimiser, 0.5)
    assert current_learning_rate(optimiser) == pytest.approx(0.05)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_emulator_rollout_feeds_back_its_own_predictions():
    inputs = np.ones((5, 7, 2, 2), dtype="float32")
    state = np.ones((5, 6, 2, 2), dtype="float32")
    norm = fit_channel_normalisation(inputs, state, np.array([0, 1, 2]))

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x[:, :6] + 1.0

    emulator = StateCNNEmulator(AddOne(), norm, torch.device("cpu"))
    pred = emulator.rollout(state[:2], np.zeros((5, 2, 2), dtype="float32"), 3)

    assert pred.shape == (3, 6, 2, 2)
    np.testing.assert_allclose(pred[:, 0, 0, 0], [2.0, 3.0, 4.0])


def test_run_experiment_smoke_writes_log_results_and_snapshot(tmp_path):
    config = write_config(tmp_path)

    result = run_experiment(config, smoke_test=True)

    log = json.loads(config.experiment_log.read_text())
    assert result["epochs_completed"] == 1
    assert len(log["rollouts"]) == 1
    rollout = log["rollouts"][0]
    assert len(rollout["eval_mse_by_time"]) == 3
    assert np.isfinite(rollout["eval_mse_mean"])
    assert "velocity_mse_mean" in rollout
    assert (tmp_path / "rollout" / "tiny-latest-surface-thickness.nc").exists()
    assert config.results_csv.exists()


def test_run_experiment_full_saves_rollout_netcdf_when_enabled(tmp_path):
    config = write_config(tmp_path, save_outputs=True)

    result = run_experiment(config)

    log = json.loads(config.experiment_log.read_text())
    assert [r["epoch"] for r in log["rollouts"]] == [1, 2]
    assert log["epochs"][1]["learning_rate"] == pytest.approx(0.0005)
    with xr.open_dataset(result["rollout_path"]) as ds:
        assert ds["rollout_layer_thickness"].shape == ds["truth_layer_thickness"].shape
        assert "rollout_meridional_velocity_centered" in ds
