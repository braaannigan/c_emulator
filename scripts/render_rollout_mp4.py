from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
FIGS_DIR = REPO_ROOT / "figs"


@dataclass(frozen=True)
class RolloutData:
    x: np.ndarray
    y: np.ndarray
    time_days: np.ndarray
    truth: np.ndarray
    rollout: np.ndarray
    vmin: float
    vmax: float


def _resolve_surface_thickness_arrays(dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    truth_name = "truth_surface_thickness" if "truth_surface_thickness" in dataset.data_vars else "truth_layer_thickness"
    rollout_name = "rollout_surface_thickness" if "rollout_surface_thickness" in dataset.data_vars else "rollout_layer_thickness"
    truth = dataset[truth_name]
    rollout = dataset[rollout_name]
    if "layers" in truth.dims:
        truth = truth.isel(layers=0)
    if "layers" in rollout.dims:
        rollout = rollout.isel(layers=0)
    return truth, rollout


def load_rollout_data(rollout_path: Path) -> RolloutData:
    with xr.open_dataset(rollout_path) as dataset:
        truth, rollout = _resolve_surface_thickness_arrays(dataset)
        if "time_days" not in truth.dims:
            time_value = float(np.asarray(dataset.coords["time_days"].values).reshape(-1)[0])
            truth = truth.expand_dims(time_days=[time_value])
        if "time_days" not in rollout.dims:
            time_value = float(np.asarray(dataset.coords["time_days"].values).reshape(-1)[0])
            rollout = rollout.expand_dims(time_days=[time_value])
        x = np.asarray(truth.coords["x"].values)
        y = np.asarray(truth.coords["y"].values)
        truth_values = np.asarray(truth.values)
        rollout_values = np.asarray(rollout.values)
        finite = np.concatenate(
            [
                truth_values[np.isfinite(truth_values)],
                rollout_values[np.isfinite(rollout_values)],
            ]
        )
        if finite.size == 0:
            raise ValueError(f"No finite values found in {rollout_path}")
        vmin = float(finite.min())
        vmax = float(finite.max())
        time_days = np.asarray(truth.coords["time_days"].values, dtype=float).reshape(-1)
    return RolloutData(
        x=x,
        y=y,
        time_days=time_days,
        truth=truth_values,
        rollout=rollout_values,
        vmin=vmin,
        vmax=vmax,
    )


def results_csv_for_family(family: str) -> Path:
    return DATA_ROOT / family / "results.csv"


def log_path_for_experiment(family: str, experiment_id: str) -> Path:
    return DATA_ROOT / family / "logs" / f"{experiment_id}.json"


def load_log(log_path: Path) -> dict:
    return json.loads(log_path.read_text(encoding="utf-8"))


def best_completed_experiment_id(family: str) -> str:
    results_path = results_csv_for_family(family)
    if not results_path.exists():
        raise FileNotFoundError(f"No results CSV for family {family}: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("epochs_completed", "0") or 0) > 1]
    if not rows:
        raise ValueError(f"No completed runs in {results_path}")
    best = min(rows, key=lambda row: float(row["eval_mse_mean"]))
    return str(best["experiment_id"])


def latest_full_rollout_path(log: dict, *, family: str, experiment_id: str) -> Path:
    rollouts = log.get("rollouts", [])
    if not isinstance(rollouts, list):
        raise ValueError(f"Experiment {experiment_id} has no rollout records")
    for rollout in reversed(rollouts):
        if not isinstance(rollout, dict):
            continue
        artifact = rollout.get("rollout_artifact")
        if isinstance(artifact, dict) and artifact.get("saved") and artifact.get("path"):
            path = Path(str(artifact["path"]))
            if path.exists():
                return path
        if rollout.get("rollout_path"):
            path = Path(str(rollout["rollout_path"]))
            if path.exists():
                return path
    latest_surface = None
    for rollout in reversed(rollouts):
        if isinstance(rollout, dict) and rollout.get("latest_surface_thickness_path"):
            latest_surface = Path(str(rollout["latest_surface_thickness_path"]))
            break
    if latest_surface is not None:
        raise FileNotFoundError(
            f"Experiment {experiment_id} in family {family} only has a final surface snapshot "
            f"({latest_surface}), not a full rollout cube. Rerun it with rollout.save_outputs: true."
        )
    raise FileNotFoundError(f"No full rollout artifact found for {experiment_id}")


def resolve_rollout_path(*, family: str, experiment_id: str | None, rollout_path: Path | None) -> tuple[str, Path]:
    if rollout_path is not None:
        return (experiment_id or rollout_path.stem, rollout_path)
    resolved_experiment_id = experiment_id or best_completed_experiment_id(family)
    log_path = log_path_for_experiment(family, resolved_experiment_id)
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log for experiment {resolved_experiment_id}: {log_path}")
    log = load_log(log_path)
    return resolved_experiment_id, latest_full_rollout_path(log, family=family, experiment_id=resolved_experiment_id)


def current_branch_name() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    branch = result.stdout.strip()
    return branch or None


def sanitise_filename_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value.strip())
    return safe.strip("-") or "rollout"


def default_output_path(experiment_id: str) -> Path:
    branch_name = current_branch_name()
    stem = sanitise_filename_component(branch_name or experiment_id)
    return FIGS_DIR / f"{stem}.mp4"


def render_rollout_mp4(rollout_path: Path, output_path: Path, *, fps: int, dpi: int) -> None:
    data = load_rollout_data(rollout_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 9), dpi=dpi, constrained_layout=True)
    truth_ax, rollout_ax = axes
    extent = [float(data.x.min()), float(data.x.max()), float(data.y.min()), float(data.y.max())]
    truth_im = truth_ax.imshow(
        data.truth[0],
        origin="lower",
        extent=extent,
        vmin=data.vmin,
        vmax=data.vmax,
        cmap="RdBu_r",
        aspect="equal",
    )
    rollout_im = rollout_ax.imshow(
        data.rollout[0],
        origin="lower",
        extent=extent,
        vmin=data.vmin,
        vmax=data.vmax,
        cmap="RdBu_r",
        aspect="equal",
    )
    truth_ax.set_title("Truth")
    rollout_ax.set_title("Rollout")
    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(rollout_im, ax=axes, shrink=0.9, pad=0.02, label="surface thickness")
    title = fig.suptitle("")

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=None) as writer:
        for time_index, time_value in enumerate(data.time_days):
            truth_im.set_data(data.truth[time_index])
            rollout_im.set_data(data.rollout[time_index])
            title.set_text(f"Surface thickness truth vs rollout at year {float(time_value) / 365.0:.2f}")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a rollout NetCDF as an MP4 animation.")
    parser.add_argument("--family", default="double_gyre_shifting_wind_2layer", help="Experiment family under data/.")
    parser.add_argument("--experiment-id", help="Experiment id to resolve from data/<family>/logs.")
    parser.add_argument("--rollout-path", type=Path, help="Direct path to a full rollout NetCDF file.")
    parser.add_argument("--output", type=Path, help="Output MP4 path. Defaults to figs/<current-branch>.mp4.")
    parser.add_argument("--fps", type=int, default=4, help="Video frame rate.")
    parser.add_argument("--dpi", type=int, default=140, help="Matplotlib render DPI.")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper
    args = parse_args()
    experiment_id, rollout_path = resolve_rollout_path(
        family=args.family,
        experiment_id=args.experiment_id,
        rollout_path=args.rollout_path,
    )
    output_path = args.output or default_output_path(experiment_id)
    render_rollout_mp4(rollout_path, output_path, fps=args.fps, dpi=args.dpi)
    print(f"experiment_id: {experiment_id}")
    print(f"rollout_path: {rollout_path}")
    print(f"output_mp4: {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
