from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, load_config
from src.data_state import RolloutWindowDataset, dgsw2l_pair_split, fit_channel_normalisation, fit_increment_normalisation, load_two_layer_state_arrays
from src.evaluation import N_LAYERS, STATE_CHANNELS, evaluate_rollout
from src.models.baseline_cnn import StateCNN
from src.models.stepper import StateStepper, StepperEmulator
from src.models.unet import StateUNet
from src.results import append_result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json_log(path: Path, log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2), encoding="utf-8")


def current_learning_rate(optimiser: torch.optim.Optimizer) -> float:
    return float(optimiser.param_groups[0]["lr"])


def decay_learning_rate(optimiser: torch.optim.Optimizer, factor: float) -> None:
    for group in optimiser.param_groups:
        group["lr"] = float(group["lr"]) * factor


def build_state_model(model_cfg: dict[str, Any], in_channels: int, out_channels: int) -> nn.Module:
    """Build the configured dgsw2l state model (``state_cnn`` or ``unet``)."""

    model_type = str(model_cfg.get("type", "state_cnn"))
    padding_mode = str(model_cfg.get("padding_mode", "zeros"))
    if model_type == "unet":
        return StateUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(model_cfg.get("hidden_channels", 32)),
            depth=int(model_cfg.get("depth", 3)),
            padding_mode=padding_mode,
        )
    if model_type != "state_cnn":
        raise ValueError(f"unknown model type {model_type!r}")
    return StateCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        hidden_layers=int(model_cfg.get("hidden_layers", 2)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        padding_mode=padding_mode,
    )


def fixed_cell_mask(state: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cells whose every state channel is constant over the training split (land), and their values."""

    train_states = state[np.union1d(train_indices, train_indices + 1)]
    mask = (train_states.min(axis=0) == train_states.max(axis=0)).all(axis=0)
    return mask, train_states[0].astype("float32")


def static_context(static_fields: list[str], state: np.ndarray, train_indices: np.ndarray, input_mean: np.ndarray, input_std: np.ndarray) -> np.ndarray | None:
    """Build normalised time-invariant input channels from the training split.

    - ``train_mean_state``: time-mean of the training states, normalised like the state inputs;
    - ``coordinates``: y and x scaled to [-1, 1];
    - ``land_mask``: 1 on cells that are constant over the training split, else 0.
    """

    channels: list[np.ndarray] = []
    n_state = state.shape[1]
    ny, nx = state.shape[2:]
    for name in static_fields:
        if name == "train_mean_state":
            mean_state = state[train_indices].mean(axis=0, dtype=np.float64)
            channels.append((mean_state - input_mean[:n_state, None, None]) / input_std[:n_state, None, None])
        elif name == "coordinates":
            yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, ny), np.linspace(-1.0, 1.0, nx), indexing="ij")
            channels.append(np.stack([yy, xx]))
        elif name == "land_mask":
            channels.append(fixed_cell_mask(state, train_indices)[0][None].astype("float32"))
        else:
            raise ValueError(f"unknown static field {name!r}")
    return np.concatenate(channels).astype("float32") if channels else None


def build_stepper(model_cfg: dict[str, Any], inputs: np.ndarray, state: np.ndarray, train_indices: np.ndarray, static_fields: list[str] | None = None) -> StateStepper:
    """Wrap the configured network in a physical-units stepper with the configured target."""

    target_mode = str(model_cfg.get("target_mode", "next_state"))
    norm = fit_channel_normalisation(inputs, state, train_indices)
    if target_mode == "residual":
        output_mean, output_std = fit_increment_normalisation(state, train_indices)
    else:
        output_mean, output_std = norm.target_mean, norm.target_std
    static = static_context(static_fields or [], state, train_indices, norm.input_mean, norm.input_std)
    n_static = 0 if static is None else static.shape[0]
    fixed_mask, fixed_values = fixed_cell_mask(state, train_indices) if model_cfg.get("fixed_land_cells", False) else (None, None)
    net = build_state_model(model_cfg, inputs.shape[1] + n_static, state.shape[1])
    return StateStepper(
        net,
        norm.input_mean,
        norm.input_std,
        output_mean,
        output_std,
        target_mode=target_mode,
        mass_conserving=bool(model_cfg.get("mass_conserving", False)),
        static=static,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )


def unrolled_loss(stepper: StateStepper, states: torch.Tensor, winds: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
    """Mean over unrolled steps of the MSE in output-normalised units.

    ``noise_std`` > 0 perturbs the window's start state with white noise of
    ``noise_std`` output-std units on wet cells, while the targets stay the
    true states, so the model learns to damp its own small-scale errors.
    """

    current = states[:, 0]
    if noise_std > 0:
        noise = torch.randn_like(current) * stepper.output_std * noise_std
        if stepper.fixed_mask.numel():
            noise = noise.masked_fill(stepper.fixed_mask, 0.0)
        current = current + noise
    steps = winds.shape[1]
    loss = states.new_zeros(())
    for k in range(steps):
        current = stepper(current, winds[:, k])
        loss = loss + (((current - states[:, k + 1]) / stepper.output_std) ** 2).mean()
    return loss / steps


def train_one_epoch(
    stepper: StateStepper,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
    noise_std: float = 0.0,
) -> float:
    stepper.train()
    losses: list[float] = []
    for states, winds in loader:
        states = states.to(device)
        winds = winds.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = unrolled_loss(stepper, states, winds, noise_std)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(stepper.parameters(), gradient_clip_norm)
        optimiser.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def rollout_diagnostics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Physical-fidelity diagnostics: spatial thickness-anomaly std and KE ratios (prediction / truth)."""

    def h_std(a: np.ndarray) -> np.ndarray:
        return a[:, :N_LAYERS].std(axis=(2, 3)).mean(axis=1)

    def ke(a: np.ndarray) -> np.ndarray:
        return 0.5 * (a[:, N_LAYERS:STATE_CHANNELS].astype("float64") ** 2).mean(axis=(1, 2, 3))

    std_ratio = h_std(prediction) / h_std(truth)
    ke_ratio = ke(prediction) / ke(truth)
    return {
        "thickness_std_ratio_mean": float(std_ratio.mean()),
        "thickness_std_ratio_last": float(std_ratio[-1]),
        "kinetic_energy_ratio_mean": float(ke_ratio.mean()),
        "kinetic_energy_ratio_last": float(ke_ratio[-1]),
    }


def save_latest_state_surface_thickness_netcdf(path: Path, truth: np.ndarray, rollout: np.ndarray, source: xr.Dataset, eval_indices: np.ndarray, experiment_id: str, epoch: int) -> None:
    """Overwrite a compact NetCDF snapshot for final-timestep surface thickness."""

    final_time_index = int(eval_indices[truth.shape[0] - 1] + 1)
    ds = xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("y", "x"), truth[-1, 0].astype("float32")),
            "rollout_surface_thickness": (("y", "x"), rollout[-1, 0].astype("float32")),
        },
        coords={
            "time_days": float(source["time_days"].values[final_time_index]),
            "y": source["y"].values,
            "x": source["x"].values,
        },
        attrs={"experiment_id": experiment_id, "epoch": epoch, "description": "Last timestep of the latest scheduled rollout; overwritten each rollout."},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def save_state_rollout_netcdf(path: Path, truth: np.ndarray, rollout: np.ndarray, source: xr.Dataset, eval_indices: np.ndarray, experiment_id: str, epoch: int) -> None:
    steps = truth.shape[0]
    time_indices = eval_indices[:steps] + 1
    n_layers = 2
    dims = ("time_days", "layers", "y", "x")
    ds = xr.Dataset(
        data_vars={
            "truth_layer_thickness": (dims, truth[:, 0:n_layers]),
            "rollout_layer_thickness": (dims, rollout[:, 0:n_layers]),
            "truth_zonal_velocity_centered": (dims, truth[:, n_layers : 2 * n_layers]),
            "rollout_zonal_velocity_centered": (dims, rollout[:, n_layers : 2 * n_layers]),
            "truth_meridional_velocity_centered": (dims, truth[:, 2 * n_layers : 3 * n_layers]),
            "rollout_meridional_velocity_centered": (dims, rollout[:, 2 * n_layers : 3 * n_layers]),
        },
        coords={
            "time_days": source["time_days"].values[time_indices],
            "layers": source["layers"].values[:n_layers],
            "y": source["y"].values,
            "x": source["x"].values,
        },
        attrs={"experiment_id": experiment_id, "epoch": epoch},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def run_experiment(config: ExperimentConfig, smoke_test: bool = False) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(str(config.raw["runtime"].get("device", "auto")))
    inputs, state, source = load_two_layer_state_arrays(config.training_data)
    forcing = inputs[:, STATE_CHANNELS]
    train_indices, eval_indices = dgsw2l_pair_split(len(state), config.train_fraction)
    unroll_steps = int(config.raw.get("training", {}).get("unroll_steps", 1))
    noise_std = float(config.raw.get("training", {}).get("input_noise_std", 0.0))
    dataset = RolloutWindowDataset(state, forcing, train_indices, unroll_steps)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model_cfg = config.raw.get("model", {})
    static_fields = [str(name) for name in config.raw.get("inputs", {}).get("static_fields", [])]
    stepper = build_stepper(model_cfg, inputs, state, train_indices, static_fields).to(device)
    optimiser = torch.optim.Adam(stepper.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epochs = config.smoke_test_epochs if smoke_test else config.epochs
    rollout_steps = 3 if smoke_test else config.rollout_steps
    eval_interval = 1 if smoke_test else config.eval_interval_epochs
    log: dict[str, Any] = {
        "metadata": {
            "experiment_id": config.experiment_id,
            "hypothesis": config.hypothesis,
            "family": config.raw["experiment"].get("family"),
            "acronym": config.raw["experiment"].get("acronym"),
            "device": str(device),
            "training_data": str(config.training_data),
            "rollout_dir": str(config.rollout_dir),
            "epochs_requested": epochs,
            "lr_decay_milestones": config.lr_decay_milestones,
            "lr_decay_factor": config.lr_decay_factor,
            "gradient_clip_norm": config.gradient_clip_norm,
            "weight_decay": config.weight_decay,
            "model_type": str(model_cfg.get("type", "state_cnn")),
            "target_mode": stepper.target_mode,
            "mass_conserving": stepper.mass_conserving,
            "unroll_steps": unroll_steps,
            "input_noise_std": noise_std,
            "static_fields": static_fields,
            "fixed_land_cells": bool(stepper.fixed_mask.numel()),
            "primary_metric": "layer-thickness MSE of fully autoregressive rollout (src/evaluation.py)",
        },
        "epochs": [],
        "rollouts": [],
        "last_update_time": utc_now(),
    }
    write_json_log(config.experiment_log, log)
    last_metrics: tuple[float, float] | None = None
    last_path: Path | None = None
    emulator = StepperEmulator(stepper, device)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss = train_one_epoch(stepper, loader, optimiser, device, gradient_clip_norm=config.gradient_clip_norm, noise_std=noise_std)
        elapsed = time.perf_counter() - started
        log["epochs"].append(
            {
                "epoch": epoch,
                "training_loss": loss,
                "duration_seconds": elapsed,
                "learning_rate": current_learning_rate(optimiser),
            }
        )
        if epoch in config.lr_decay_milestones:
            decay_learning_rate(optimiser, config.lr_decay_factor)
        if epoch % eval_interval == 0 or epoch == epochs:
            result = evaluate_rollout(emulator, state, forcing, eval_indices, rollout_steps)
            suffix = "smoke" if smoke_test else f"epoch{epoch:04d}"
            artifact = {
                "saved": False,
                "format": "not_saved",
                "reason": "derived rollout arrays are reproducible from code, config, data, and logged metrics",
            }
            path = None
            if config.save_rollout_outputs:
                path = config.rollout_dir / f"{config.experiment_id}-{suffix}.nc"
                save_state_rollout_netcdf(path, result.truth, result.prediction, source, eval_indices, config.experiment_id, epoch)
                artifact = {"saved": True, "format": "netcdf", "path": str(path)}
            latest_surface_path = config.rollout_dir / f"{config.experiment_id}-latest-surface-thickness.nc"
            save_latest_state_surface_thickness_netcdf(latest_surface_path, result.truth, result.prediction, source, eval_indices, config.experiment_id, epoch)
            log["rollouts"].append(
                {
                    "epoch": epoch,
                    "eval_mse_by_time": [float(v) for v in result.thickness_mse_by_time],
                    "eval_mse_mean": result.eval_mse_mean,
                    "eval_mse_last": result.eval_mse_last,
                    "velocity_mse_by_time": [float(v) for v in result.velocity_mse_by_time],
                    "velocity_mse_mean": result.velocity_mse_mean,
                    **rollout_diagnostics(result.truth, result.prediction),
                    "rollout_artifact": artifact,
                    "latest_surface_thickness_path": str(latest_surface_path),
                    **({"rollout_path": str(path)} if path is not None else {}),
                }
            )
            last_metrics = (result.eval_mse_mean, result.eval_mse_last)
            last_path = path
        log["last_update_time"] = utc_now()
        write_json_log(config.experiment_log, log)
    if last_metrics is None:  # pragma: no cover - the final epoch always triggers a rollout
        raise RuntimeError("No rollout was produced")
    if not smoke_test:
        checkpoint = config.results_csv.parent / "checkpoints" / f"{config.experiment_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stepper.state_dict(), checkpoint)
    append_result(config.results_csv, config.experiment_id, last_metrics[0], last_metrics[1], epochs, config.hypothesis, config.conclusion)
    return {"experiment_id": config.experiment_id, "eval_mse_mean": last_metrics[0], "eval_mse_last": last_metrics[1], "epochs_completed": epochs, "rollout_path": None if last_path is None else str(last_path), "experiment_log": str(config.experiment_log)}


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Run dgsw2l baseline state emulator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    result = run_experiment(load_config(args.config), args.smoke_test)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
