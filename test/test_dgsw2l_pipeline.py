import json

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from src.config import load_config
from src.models.baseline_cnn import StateCNN
from src.pipelines.dgsw2l_baseline import (
    build_state_model,
    build_stepper,
    rollout_diagnostics,
    fixed_cell_mask,
    static_context,
    unrolled_loss,
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
    assert convs[0].padding_mode == "zeros"
    reflect = build_state_model({"padding_mode": "reflect"}, 7, 6)
    assert all(layer.padding_mode == "reflect" for layer in reflect.net if isinstance(layer, torch.nn.Conv2d))


def test_learning_rate_helpers_and_device():
    optimiser = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    decay_learning_rate(optimiser, 0.5)
    assert current_learning_rate(optimiser) == pytest.approx(0.05)
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_build_stepper_selects_residual_increment_normalisation():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(6, 6, 3, 3)).astype("float32")
    inputs = np.concatenate([state, np.zeros((6, 1, 3, 3), dtype="float32")], axis=1)
    idx = np.arange(4)
    residual = build_stepper({"target_mode": "residual", "mass_conserving": True, "hidden_channels": 2, "hidden_layers": 1}, inputs, state, idx)
    full = build_stepper({"hidden_channels": 2, "hidden_layers": 1}, inputs, state, idx)
    delta = state[idx + 1] - state[idx]
    np.testing.assert_allclose(residual.output_std.flatten().numpy(), delta.std(axis=(0, 2, 3)), rtol=1e-5)
    np.testing.assert_allclose(full.output_std.flatten().numpy(), state[idx + 1].std(axis=(0, 2, 3)), rtol=1e-5)
    assert residual.mass_conserving and residual.target_mode == "residual"
    assert full.target_mode == "next_state" and not full.mass_conserving


def test_unrolled_loss_averages_over_steps_and_feeds_back_predictions():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x[:, :6])

    from src.models.stepper import StateStepper

    stepper = StateStepper(Identity(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual")
    states = torch.zeros(1, 3, 6, 2, 2)
    states[:, 2] = 2.0
    winds = torch.zeros(1, 2, 2, 2)
    # Identity residual predicts the start state (0); errors are 0 then 2 -> mean (0 + 4) / 2.
    assert float(unrolled_loss(stepper, states, winds)) == pytest.approx(2.0)


def test_rollout_diagnostics_ratios():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(3, 6, 4, 4))
    diag = rollout_diagnostics(truth, truth * 2.0)
    assert diag["thickness_std_ratio_mean"] == pytest.approx(2.0)
    assert diag["thickness_std_ratio_last"] == pytest.approx(2.0)
    assert diag["kinetic_energy_ratio_mean"] == pytest.approx(4.0)
    assert diag["kinetic_energy_ratio_last"] == pytest.approx(4.0)


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
    assert "thickness_std_ratio_mean" in rollout
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


def test_build_stepper_nudging_uses_training_mean_climate():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(6, 6, 4, 4)).astype("float32")
    inputs = np.concatenate([state, np.zeros((6, 1, 4, 4), dtype="float32")], axis=1)
    stepper = build_stepper({"hidden_channels": 2, "hidden_layers": 1, "target_mode": "residual", "nudge_rate": 0.1, "nudge_pool": 2}, inputs, state, np.arange(4))
    assert stepper.nudge_rate == 0.1 and stepper.nudge_pool == 2
    np.testing.assert_allclose(stepper.climate[0].numpy(), state[:4].mean(axis=0), rtol=1e-5)
    plain = build_stepper({"hidden_channels": 2, "hidden_layers": 1}, inputs, state, np.arange(4))
    assert plain.nudge_rate == 0.0 and plain.climate.numel() == 0


def test_run_experiment_residual_mass_conserving_unrolled(tmp_path):
    config = write_config(tmp_path)
    config.raw["model"].update({"target_mode": "residual", "mass_conserving": True, "nudge_rate": 0.05, "nudge_pool": 2})
    config.raw["training"]["unroll_steps"] = 2

    run_experiment(config, smoke_test=True)

    log = json.loads(config.experiment_log.read_text())
    assert log["metadata"]["unroll_steps"] == 2
    assert log["metadata"]["nudge_rate"] == 0.05
    assert log["metadata"]["target_mode"] == "residual"
    assert log["metadata"]["mass_conserving"] is True


def test_static_context_channels_and_stepper_input_width():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(6, 6, 3, 4)).astype("float32")
    inputs = np.concatenate([state, np.zeros((6, 1, 3, 4), dtype="float32")], axis=1)
    idx = np.arange(4)
    mean, std = np.zeros(7, dtype="float32"), np.full(7, 2.0, dtype="float32")
    static = static_context(["train_mean_state", "coordinates"], state, idx, mean, std)
    assert static.shape == (8, 3, 4)
    np.testing.assert_allclose(static[:6], state[idx].mean(axis=0) / 2.0, rtol=1e-5)
    np.testing.assert_allclose(static[6, :, 0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(static[7, 0, [0, -1]], [-1.0, 1.0])
    assert static_context([], state, idx, mean, std) is None
    with pytest.raises(ValueError):
        static_context(["bogus"], state, idx, mean, std)
    stepper = build_stepper({"hidden_channels": 2, "hidden_layers": 1, "target_mode": "residual"}, inputs, state, idx, ["coordinates"])
    assert stepper.net.net[0].in_channels == 9
    assert stepper(torch.zeros(2, 6, 3, 4), torch.zeros(2, 3, 4)).shape == (2, 6, 3, 4)


def test_run_experiment_full_saves_checkpoint(tmp_path):
    config = write_config(tmp_path)
    config.raw["inputs"] = {"static_fields": ["coordinates"]}

    run_experiment(config)

    state_dict = torch.load(tmp_path / "checkpoints" / "tiny.pt")
    assert state_dict["static"].shape[:2] == (1, 2)


def test_fixed_cell_mask_and_land_mask_channel():
    rng = np.random.default_rng(0)
    state = rng.normal(size=(6, 6, 3, 4)).astype("float32")
    state[:, :, 0, :] = 7.0
    idx = np.arange(4)
    mask, values = fixed_cell_mask(state, idx)
    assert mask[0].all() and not mask[1:].any()
    np.testing.assert_allclose(values[:, 0], 7.0)
    static = static_context(["land_mask"], state, idx, np.zeros(7, dtype="float32"), np.ones(7, dtype="float32"))
    np.testing.assert_allclose(static[0], mask.astype("float32"))
    inputs = np.concatenate([state, np.zeros((6, 1, 3, 4), dtype="float32")], axis=1)
    stepper = build_stepper({"hidden_channels": 2, "hidden_layers": 1, "target_mode": "residual", "fixed_land_cells": True}, inputs, state, idx)
    assert stepper.fixed_mask.shape == (1, 1, 3, 4)


def test_unrolled_loss_noise_perturbs_wet_cells_only():
    from src.models.stepper import StateStepper

    class Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x[:, :6])

    mask = np.ones((2, 2), dtype=bool)
    pinned = StateStepper(Zero(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual", fixed_mask=mask, fixed_values=np.zeros((6, 2, 2)))
    free = StateStepper(Zero(), np.zeros(7), np.ones(7), np.zeros(6), np.ones(6), target_mode="residual")
    states = torch.zeros(4, 2, 6, 2, 2)
    winds = torch.zeros(4, 1, 2, 2)
    assert float(unrolled_loss(pinned, states, winds, noise_std=1.0)) == 0.0
    torch.manual_seed(0)
    assert float(unrolled_loss(free, states, winds, noise_std=1.0)) == pytest.approx(1.0, rel=0.5)


def test_run_experiment_logs_noise_std(tmp_path):
    config = write_config(tmp_path)
    config.raw["training"]["input_noise_std"] = 0.5

    run_experiment(config, smoke_test=True)

    assert json.loads(config.experiment_log.read_text())["metadata"]["input_noise_std"] == 0.5


def test_build_state_model_unet_and_unknown_type():
    from src.models.unet import StateUNet

    model = build_state_model({"type": "unet", "hidden_channels": 4, "depth": 2}, 7, 6)
    assert isinstance(model, StateUNet)
    assert model(torch.zeros(1, 7, 8, 8)).shape == (1, 6, 8, 8)
    with pytest.raises(ValueError):
        build_state_model({"type": "bogus"}, 7, 6)
