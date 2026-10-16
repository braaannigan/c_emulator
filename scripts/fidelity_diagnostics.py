"""Statistical-fidelity diagnostics for a trained dgsw2l emulator.

Usage:
    uv run python scripts/fidelity_diagnostics.py --config <config> [--checkpoint <pt>] [--long-years 25]

1. Held-out period: the fixed evaluator's rollout (one free run from the
   split point) is compared with the truth over the same 143 steps.
2. Long control run: a free run from the last state of the record, forced by
   the analytic wind for ``--long-years`` further years, is compared with the
   statistics of the full 715-step record, and split into yearly blocks to
   expose drift. The free run starts from the last true state of the record;
   this uses held-out truth only as an initial condition for diagnostics,
   never in the benchmark.

Writes ``data/double_gyre_shifting_wind_2layer/diagnostics/<id>/report.json``
and figures (spectra, EKE and variance maps, autocorrelation, drift) there.
Inference only; runs on the local machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data_state import dgsw2l_pair_split, load_two_layer_state_arrays  # noqa: E402
from src.diagnostics import (  # noqa: E402
    INTERIOR,
    autocorrelation,
    block_statistics,
    compare_statistics,
    spread_skill,
    eke_map,
    extended_forcing,
    kinetic_energy_series,
    variance_map,
    wavenumber_spectrum,
)
from src.evaluation import N_LAYERS, STATE_CHANNELS, evaluate_rollout  # noqa: E402
from src.scorecard import drift_score, scorecard  # noqa: E402
from src.models.stepper import StepperEmulator  # noqa: E402
from src.pipelines.dgsw2l_baseline import build_stepper, resolve_device  # noqa: E402

STEPS_PER_YEAR = 52


def load_emulator(config, inputs, state, train_indices, checkpoint: Path, device: torch.device):
    model_cfg = config.raw.get("model", {})
    static_fields = [str(name) for name in config.raw.get("inputs", {}).get("static_fields", [])]
    model = build_stepper(model_cfg, inputs, state, train_indices, static_fields)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    return StepperEmulator(model, device)


def plot_spectra(path: Path, pairs: dict[str, tuple[np.ndarray, np.ndarray]], title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, layer in enumerate(range(N_LAYERS)):
        for col, (name, index) in enumerate((("thickness", layer), ("zonal velocity", N_LAYERS + layer))):
            ax = axes[row, col]
            for label, (model, truth) in pairs.items():
                ys, xs = INTERIOR
                k, pm = wavenumber_spectrum(model[:, index, ys, xs], "x")
                _, pt = wavenumber_spectrum(truth[:, index, ys, xs], "x")
                ax.loglog(k[1:], pt[1:], "k-", lw=2, label=f"truth ({label})" if label == list(pairs)[0] else None)
                ax.loglog(k[1:], pm[1:], "--", label=f"emulator ({label})")
            ax.set_title(f"layer {layer} {name}: zonal wavenumber spectrum")
            ax.set_xlabel("cycles per cell (10 km)")
            ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_maps(path: Path, model: np.ndarray, truth: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 9))
    for layer in range(N_LAYERS):
        u, v, h = N_LAYERS + layer, 2 * N_LAYERS + layer, layer
        for col, (name, fm, ft) in enumerate(
            (
                ("EKE", eke_map(model[:, u], model[:, v]), eke_map(truth[:, u], truth[:, v])),
                ("h variance", variance_map(model[:, h]), variance_map(truth[:, h])),
            )
        ):
            vmax = float(np.percentile(ft, 99))
            for offset, (label, field) in enumerate((("truth", ft), ("emulator", fm))):
                ax = axes[layer, 2 * col + offset]
                image = ax.imshow(field, origin="lower", vmin=0, vmax=vmax, cmap="magma")
                ax.set_title(f"L{layer} {name}: {label}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_autocorrelation(path: Path, model: np.ndarray, truth: np.ndarray, title: str, max_lag: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ys, xs = INTERIOR
    for layer in range(N_LAYERS):
        ax.plot(autocorrelation(truth[:, layer, ys, xs], max_lag), "-", label=f"truth L{layer}")
        ax.plot(autocorrelation(model[:, layer, ys, xs], max_lag), "--", label=f"emulator L{layer}")
    ax.axhline(np.exp(-1), color="grey", lw=0.8)
    ax.set_xlabel("lag (7-day steps)")
    ax.set_ylabel("thickness autocorrelation")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_drift(path: Path, long_run: np.ndarray, truth: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    years = np.arange(len(long_run)) / STEPS_PER_YEAR
    ke_model, ke_truth = kinetic_energy_series(long_run), kinetic_energy_series(truth)
    for layer in range(N_LAYERS):
        axes[0].plot(years, ke_model[:, layer], lw=0.8, label=f"emulator L{layer}")
        axes[0].axhspan(np.percentile(ke_truth[:, layer], 5), np.percentile(ke_truth[:, layer], 95), alpha=0.15, label=f"truth 5-95% L{layer}")
    axes[0].set_ylabel("domain-mean KE (m^2/s^2)")
    axes[0].legend(fontsize=8)
    blocks = block_statistics(long_run, STEPS_PER_YEAR)
    truth_blocks = block_statistics(truth, STEPS_PER_YEAR)
    axes[1].plot(np.arange(len(blocks["thickness_std"])) + 0.5, blocks["thickness_std"], "o-", label="emulator (yearly)")
    axes[1].axhspan(min(truth_blocks["thickness_std"]), max(truth_blocks["thickness_std"]), alpha=0.15, label="truth yearly range")
    axes[1].set_ylabel("thickness std (m)")
    axes[1].set_xlabel("years of free run")
    axes[1].legend(fontsize=8)
    fig.suptitle("Long control run: drift check")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical-fidelity diagnostics for a trained dgsw2l emulator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--long-years", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--members", type=int, default=4, help="Ensemble members for the spread-skill check of stochastic models")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    experiment_id = config.experiment_id
    checkpoint = Path(args.checkpoint) if args.checkpoint else config.results_csv.parent / "checkpoints" / f"{experiment_id}.pt"
    out_dir = config.results_csv.parent / "diagnostics" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    inputs, state, source = load_two_layer_state_arrays(config.training_data)
    forcing = inputs[:, STATE_CHANNELS]
    train_indices, eval_indices = dgsw2l_pair_split(len(state), config.train_fraction)
    emulator = load_emulator(config, inputs, state, train_indices, checkpoint, device)
    period = float(source.attrs.get("wind_shift_period_days", 40.0))
    times = source["time_days"].values.astype("float64")

    started = time.perf_counter()
    held_out = evaluate_rollout(emulator, state, forcing, eval_indices)
    eval_times = times[eval_indices[0] + 1 : eval_indices[0] + 1 + len(held_out.prediction)]
    report = {
        "experiment_id": experiment_id,
        "checkpoint": str(checkpoint),
        "held_out": {
            "eval_mse_mean": held_out.eval_mse_mean,
            "eval_mse_last": held_out.eval_mse_last,
            "velocity_mse_mean": held_out.velocity_mse_mean,
            "statistics": compare_statistics(held_out.prediction, held_out.truth, eval_times, eval_times, period),
            "scorecard": scorecard(held_out.prediction, held_out.truth, held_out.thickness_mse_by_time),
        },
    }
    plot_spectra(out_dir / "spectra_held_out.png", {"held-out": (held_out.prediction, held_out.truth)}, f"{experiment_id}: held-out period (143 steps)")
    plot_maps(out_dir / "maps_held_out.png", held_out.prediction, held_out.truth, f"{experiment_id}: held-out period")
    plot_autocorrelation(out_dir / "autocorrelation_held_out.png", held_out.prediction, held_out.truth, "held-out period")

    if getattr(emulator.stepper, "stochastic_channels", 0):
        # Stochastic emulators: is the ensemble actually spread out (calibrated ~1)?
        start = int(eval_indices[0])
        members = []
        for member in range(args.members):
            torch.manual_seed(1000 + member)
            members.append(np.asarray(emulator.rollout(state[: start + 1], forcing[: start + 1 + len(held_out.prediction)], len(held_out.prediction)), dtype="float32"))
        report["held_out"]["ensemble"] = {"members": args.members, **spread_skill(np.stack(members), held_out.truth)}

    long_steps = int(round(args.long_years * STEPS_PER_YEAR))
    long_times, long_forcing = extended_forcing(source, long_steps)
    long_run = np.asarray(emulator.rollout(state, long_forcing, long_steps), dtype="float32")
    run_times = long_times[len(state) :]
    report["long_run"] = {
        "years": args.long_years,
        "steps": long_steps,
        "finite": bool(np.isfinite(long_run).all()),
        "statistics_vs_full_record": compare_statistics(long_run, state, run_times, times, period),
        "yearly_blocks": block_statistics(long_run, STEPS_PER_YEAR),
        "truth_yearly_blocks": block_statistics(state, STEPS_PER_YEAR),
    }
    report["long_run"]["drift"] = drift_score(report["long_run"]["yearly_blocks"]["kinetic_energy"], report["long_run"]["truth_yearly_blocks"]["kinetic_energy"])
    plot_spectra(out_dir / "spectra_long_run.png", {"long run": (long_run, state)}, f"{experiment_id}: {args.long_years:g}-year free run vs full record")
    plot_maps(out_dir / "maps_long_run.png", long_run, state, f"{experiment_id}: {args.long_years:g}-year free run vs full record")
    plot_autocorrelation(out_dir / "autocorrelation_long_run.png", long_run, state, "long free run vs full record")
    plot_drift(out_dir / "drift_long_run.png", long_run, state)
    report["runtime_seconds"] = time.perf_counter() - started

    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)
    print(f"wrote {out_dir}")


def print_summary(report: dict) -> None:
    card = report["held_out"].get("scorecard")
    if card:
        print("== scorecard (held-out, lower is better): " + ", ".join(f"{k} {v:.3g}" for k, v in card.items()))
    ensemble = report["held_out"].get("ensemble")
    if ensemble:
        print("== ensemble: " + ", ".join(f"{k} {v:.3g}" for k, v in ensemble.items()))
    drift = report["long_run"].get("drift")
    if drift:
        print("== long-run drift: " + ", ".join(f"{k} {v:.3g}" for k, v in drift.items()))
    for section, key in (("held_out", "statistics"), ("long_run", "statistics_vs_full_record")):
        stats = report[section][key]
        print(f"== {section}: KE ratio {stats['kinetic_energy_ratio']:.2f}, thickness std ratio {stats['thickness_std_ratio']:.2f}")
        for layer, entry in stats["layers"].items():
            spec = entry["thickness_spectrum_x_band_ratio"]
            uspec = entry["zonal_velocity_spectrum_x_band_ratio"]
            print(
                f"  L{layer}: h-spectrum ratio large/eddy/small {spec['large']:.2f}/{spec['eddy']:.2f}/{spec['small']:.2f};"
                f" u-spectrum {uspec['large']:.2f}/{uspec['eddy']:.2f}/{uspec['small']:.2f};"
                f" EKE map ratio {entry['eke_map']['ratio']:.2f} corr {entry['eke_map']['pattern_correlation']:.2f};"
                f" h-var map ratio {entry['thickness_variance_map']['ratio']:.2f} corr {entry['thickness_variance_map']['pattern_correlation']:.2f};"
                f" e-fold {entry['thickness_efolding_steps']['model']:.1f} vs {entry['thickness_efolding_steps']['truth']:.1f} steps;"
                f" freq slow/seasonal/fast {entry['thickness_frequency_band_ratio']['slow']:.2f}/{entry['thickness_frequency_band_ratio']['seasonal']:.2f}/{entry['thickness_frequency_band_ratio']['fast']:.2f};"
                f" wind R2 {entry['wind_response_r2']['model']:.3f} vs {entry['wind_response_r2']['truth']:.3f}"
            )
    if "yearly_blocks" in report["long_run"]:
        ke = report["long_run"]["yearly_blocks"]["kinetic_energy"]
        truth_ke = report["long_run"]["truth_yearly_blocks"]["kinetic_energy"]
        print(f"  long-run yearly KE: first {ke[0]:.5f}, last {ke[-1]:.5f}; truth yearly range {min(truth_ke):.5f}-{max(truth_ke):.5f}; finite={report['long_run']['finite']}")


if __name__ == "__main__":
    main()
