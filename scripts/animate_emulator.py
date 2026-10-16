"""Animate a trained dgsw2l emulator as an MP4.

Usage:
    uv run python scripts/animate_emulator.py --config <config> [--mode long|held-out]
        [--long-years 25] [--stride 2] [--fps 20] [--layer 0] [--output figs/<name>.mp4]

Modes:
- ``held-out``: the fixed evaluator's free run over the held-out period
  (143 weekly steps) side by side with the truth.
- ``long``: a free run from the last state of the record, forced by the
  analytic wind for ``--long-years`` years (no truth exists beyond the
  record). The KE trace is shown against the truth's 5-95% range, so drift
  is visible.

Panels show thickness anomaly (from the training-mean state) and relative
vorticity for one layer. Rollouts are generated from the checkpoint, so no
saved rollout file is needed. Inference only; runs locally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config  # noqa: E402
from src.diagnostics import INTERIOR, extended_forcing, kinetic_energy_series, relative_vorticity  # noqa: E402
from src.emulator_loading import load_trained_emulator  # noqa: E402
from src.evaluation import N_LAYERS, evaluate_rollout  # noqa: E402
from src.pipelines.dgsw2l_baseline import resolve_device  # noqa: E402

STEPS_PER_YEAR = 52


def symmetric_limit(*fields: np.ndarray, percentile: float = 99.5) -> float:
    return float(max(np.nanpercentile(np.abs(field), percentile) for field in fields))


def render(
    output: Path,
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    times_years: np.ndarray,
    ke: np.ndarray,
    ke_band: tuple[float, float],
    title: str,
    stride: int,
    fps: int,
    extent: tuple[float, float, float, float],
) -> None:
    """``panels``: (row label, thickness anomaly (t, y, x), vorticity (t, y, x)) per row."""

    rows = len(panels)
    h_lim = symmetric_limit(*(p[1] for p in panels))
    z_lim = symmetric_limit(*(p[2] for p in panels))
    fig = plt.figure(figsize=(9.5, 5.2 * rows + 2.2), dpi=100)
    grid = fig.add_gridspec(rows + 1, 2, height_ratios=[5] * rows + [1.6])
    images = []
    for row, (label, h, zeta) in enumerate(panels):
        ax_h, ax_z = fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])
        cmap_h, cmap_z = plt.get_cmap("RdBu_r").copy(), plt.get_cmap("PuOr_r").copy()
        cmap_h.set_bad("0.75")
        cmap_z.set_bad("0.75")
        im_h = ax_h.imshow(h[0], origin="lower", extent=extent, cmap=cmap_h, vmin=-h_lim, vmax=h_lim)
        im_z = ax_z.imshow(zeta[0], origin="lower", extent=extent, cmap=cmap_z, vmin=-z_lim, vmax=z_lim)
        ax_h.set_title(f"{label}: thickness anomaly (m)", fontsize=10)
        ax_z.set_title(f"{label}: relative vorticity (1/s)", fontsize=10)
        for ax in (ax_h, ax_z):
            ax.set_xlabel("x (km)")
            ax.set_ylabel("y (km)")
        fig.colorbar(im_h, ax=ax_h, fraction=0.046)
        fig.colorbar(im_z, ax=ax_z, fraction=0.046, format="%.0e")
        images.append((im_h, im_z, h, zeta))
    ax_ke = fig.add_subplot(grid[rows, :])
    ax_ke.axhspan(*ke_band, color="grey", alpha=0.2, label="truth 5-95%")
    ax_ke.plot(times_years, ke, color="tab:blue", lw=0.8, alpha=0.35)
    (trace,) = ax_ke.plot([], [], color="tab:blue", lw=1.4, label="emulator")
    marker = ax_ke.axvline(times_years[0], color="k", lw=0.8)
    ax_ke.set_xlim(times_years[0], times_years[-1])
    ax_ke.set_ylabel("domain-mean KE")
    ax_ke.set_xlabel("years")
    ax_ke.legend(loc="upper right", fontsize=8)
    suptitle = fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
        for index in range(0, len(times_years), stride):
            for im_h, im_z, h, zeta in images:
                im_h.set_data(h[index])
                im_z.set_data(zeta[index])
            trace.set_data(times_years[: index + 1], ke[: index + 1])
            marker.set_xdata([times_years[index], times_years[index]])
            suptitle.set_text(f"{title}  -  year {times_years[index]:.2f}")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a trained dgsw2l emulator as an MP4")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["long", "held-out"], default="long")
    parser.add_argument("--long-years", type=float, default=25.0)
    parser.add_argument("--layer", type=int, default=0, choices=range(N_LAYERS))
    parser.add_argument("--stride", type=int, default=None, help="Render every Nth step (default 1 held-out, 2 long)")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second (default 8 held-out, 20 long)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    loaded = load_trained_emulator(config, args.checkpoint, resolve_device(args.device))
    source, state = loaded.source, loaded.state
    dx = float(source["x"].values[1] - source["x"].values[0])
    extent = (float(source["x"].values[0]) / 1e3, float(source["x"].values[-1]) / 1e3, float(source["y"].values[0]) / 1e3, float(source["y"].values[-1]) / 1e3)
    mean_h = state[loaded.train_indices, args.layer].mean(axis=0)
    u, v = N_LAYERS + args.layer, 2 * N_LAYERS + args.layer
    truth_ke = kinetic_energy_series(state)[:, args.layer]
    ke_band = (float(np.percentile(truth_ke, 5)), float(np.percentile(truth_ke, 95)))

    # Land cells (constant over training) are greyed out; vorticity is shown on
    # the wet interior only, since differences across the land checkerboard at
    # the walls are not meaningful.
    land = (state[loaded.train_indices].min(axis=0) == state[loaded.train_indices].max(axis=0)).all(axis=0)
    wall = np.ones(land.shape, dtype=bool)
    wall[INTERIOR] = False

    def fields(run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.where(land, np.nan, run[:, args.layer] - mean_h)
        zeta = np.where(wall, np.nan, relative_vorticity(run[:, u], run[:, v], dx))
        return h, zeta

    if args.mode == "held-out":
        result = evaluate_rollout(loaded.emulator, state, loaded.forcing, loaded.eval_indices)
        start = int(loaded.eval_indices[0]) + 1
        times_years = source["time_days"].values[start : start + len(result.prediction)] / 365.25
        panels = [("truth", *fields(result.truth)), ("emulator", *fields(result.prediction))]
        ke = kinetic_energy_series(result.prediction)[:, args.layer]
        title = f"{config.experiment_id}: held-out free run, layer {args.layer}"
        stride, fps = args.stride or 1, args.fps or 8
    else:
        steps = int(round(args.long_years * STEPS_PER_YEAR))
        times, forcing = extended_forcing(source, steps)
        run = np.asarray(loaded.emulator.rollout(state, forcing, steps), dtype="float32")
        times_years = (times[len(state) :] - times[len(state) - 1]) / 365.25
        panels = [("emulator", *fields(run))]
        ke = kinetic_energy_series(run)[:, args.layer]
        title = f"{config.experiment_id}: {args.long_years:g}-year free run, layer {args.layer}"
        stride, fps = args.stride or 2, args.fps or 20
    output = args.output or ROOT / "figs" / f"{config.experiment_id}-{args.mode}-layer{args.layer}.mp4"
    render(output, panels, times_years, ke, ke_band, title, stride, fps, extent)
    print(f"wrote {output} ({len(range(0, len(times_years), stride))} frames at {fps} fps)")


if __name__ == "__main__":
    main()
