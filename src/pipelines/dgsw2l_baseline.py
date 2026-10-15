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
from src.data_state import ChannelNormalisation, NextStepStateDataset, dgsw2l_pair_split, fit_channel_normalisation, load_two_layer_state_arrays
from src.evaluation import STATE_CHANNELS, evaluate_rollout
from src.models.baseline_cnn import StateCNN
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
    """Build the configured dgsw2l state model."""

    return StateCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        hidden_layers=int(model_cfg.get("hidden_layers", 2)),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    losses: list[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimiser.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


class StateCNNEmulator:
    """Adapts a next-state CNN to the fixed evaluator's ``rollout`` interface."""

    def __init__(self, model: nn.Module, norm: ChannelNormalisation, device: torch.device):
        self.model = model
        self.norm = norm
        self.device = device

    def rollout(self, history: np.ndarray, forcing: np.ndarray, steps: int) -> np.ndarray:
        self.model.eval()
        start = len(history) - 1
        current_state = history[-1].astype("float32")
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for offset in range(steps):
                x = np.concatenate([current_state, forcing[start + offset][None]], axis=0)
                x = (x - self.norm.input_mean[:, None, None]) / self.norm.input_std[:, None, None]
                pred_norm = self.model(torch.from_numpy(x[None].astype("float32")).to(self.device)).cpu().numpy()[0]
                current_state = (pred_norm * self.norm.target_std[:, None, None] + self.norm.target_mean[:, None, None]).astype("float32")
                predictions.append(current_state)
        return np.stack(predictions)


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
    norm = fit_channel_normalisation(inputs, state, train_indices)
    dataset = NextStepStateDataset(inputs, state, train_indices, norm)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model_cfg = config.raw.get("model", {})
    model = build_state_model(model_cfg, inputs.shape[1], state.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
            "primary_metric": "layer-thickness MSE of fully autoregressive rollout (src/evaluation.py)",
        },
        "epochs": [],
        "rollouts": [],
        "last_update_time": utc_now(),
    }
    write_json_log(config.experiment_log, log)
    last_metrics: tuple[float, float] | None = None
    last_path: Path | None = None
    emulator = StateCNNEmulator(model, norm, device)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss = train_one_epoch(model, loader, optimiser, device, gradient_clip_norm=config.gradient_clip_norm)
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
