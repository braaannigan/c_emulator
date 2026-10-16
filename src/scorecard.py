"""Secondary evaluation scorecard for dgsw2l emulators.

The fixed benchmark metric ``eval_mse_mean`` (``src/evaluation.py``) is
dominated by the chaotic tail: beyond ~50 weekly steps any realistic free run
saturates near twice the climate variance, single checkpoints scatter by
~+/-20, and smoother (less realistic) fields score better. This scorecard is
computed from the same evaluator output (truth and prediction over the
held-out rollout) and adds metrics that separate models on what can be
predicted and on physical realism. It never changes the benchmark.

Lower is better for every score:

- ``predictable_mse``: mean layer-thickness MSE over the first
  ``PREDICTABLE_STEPS`` (26 weeks, ~6 months) of the rollout, the range where
  forecast skill exists.
- ``spectral_error``: mean |ln(model / truth)| power ratio in the large and
  eddy wavenumber bands of thickness and zonal velocity (both layers, x and y).
- ``noise_excess_decades``: mean log10 of the small-scale (near-grid) power
  excess over truth, floored at 0 (unphysical grid noise).
- ``eke_amplitude_error`` and ``eke_pattern_error``: |ln| of the eddy kinetic
  energy map's domain-mean ratio and 1 - its pattern correlation.
- ``realism``: ``spectral_error + eke_amplitude_error + eke_pattern_error +
  0.25 * noise_excess_decades``, a single realism number.

``drift_score`` summarises a long free run (see
``scripts/fidelity_diagnostics.py``): the fraction of years whose mean KE
falls outside the truth's yearly range, plus |ln| of the last-five-years to
first-five-years KE ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.diagnostics import INTERIOR, SPATIAL_BANDS, band_power, eke_map, pattern_stats, wavenumber_spectrum
from src.evaluation import N_LAYERS

PREDICTABLE_STEPS = 26
NOISE_WEIGHT = 0.25


def predictable_mse(thickness_mse_by_time: np.ndarray, steps: int = PREDICTABLE_STEPS) -> float:
    return float(np.mean(np.asarray(thickness_mse_by_time)[:steps]))


def _log_ratio(model: float, truth: float) -> float:
    return float(np.log(model / truth)) if model > 0 and truth > 0 else float("nan")


def _interior(field: np.ndarray) -> np.ndarray:
    """Wet interior of a dgsw2l-sized grid; tiny test grids are used whole."""

    ys, xs = INTERIOR
    return field[..., ys, xs] if field.shape[-2] > 8 and field.shape[-1] > 6 else field


def spectral_scores(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Spectral error (large + eddy bands) and small-scale noise excess, over wet interior cells."""

    model, true = _interior(prediction), _interior(truth)
    errors, excess = [], []
    for layer in range(N_LAYERS):
        for channel in (layer, N_LAYERS + layer):
            for axis in ("x", "y"):
                freq, pm = wavenumber_spectrum(model[:, channel], axis)
                _, pt = wavenumber_spectrum(true[:, channel], axis)
                bm, bt = band_power(freq, pm, SPATIAL_BANDS), band_power(freq, pt, SPATIAL_BANDS)
                errors.extend(abs(_log_ratio(bm[band], bt[band])) for band in ("large", "eddy") if bt[band] > 0)
                if bt["small"] > 0 and bm["small"] > 0:
                    excess.append(max(0.0, float(np.log10(bm["small"] / bt["small"]))))
    return {
        "spectral_error": float(np.nanmean(errors)) if errors else float("nan"),
        "noise_excess_decades": float(np.mean(excess)) if excess else float("nan"),
    }


def eke_scores(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    model, true = _interior(prediction), _interior(truth)
    amplitude, pattern = [], []
    for layer in range(N_LAYERS):
        u, v = N_LAYERS + layer, 2 * N_LAYERS + layer
        stats = pattern_stats(eke_map(model[:, u], model[:, v]), eke_map(true[:, u], true[:, v]))
        amplitude.append(abs(float(np.log(stats["ratio"]))) if stats["ratio"] > 0 else float("nan"))
        pattern.append(1.0 - stats["pattern_correlation"])
    return {"eke_amplitude_error": float(np.mean(amplitude)), "eke_pattern_error": float(np.mean(pattern))}


def scorecard(prediction: np.ndarray, truth: np.ndarray, thickness_mse_by_time: np.ndarray) -> dict[str, float]:
    """All held-out scores for one rollout (prediction and truth shaped ``(time, 6, y, x)``)."""

    scores = {"predictable_mse": predictable_mse(thickness_mse_by_time)}
    scores.update(spectral_scores(prediction, truth))
    scores.update(eke_scores(prediction, truth))
    noise = scores["noise_excess_decades"]
    scores["realism"] = float(
        scores["spectral_error"] + scores["eke_amplitude_error"] + scores["eke_pattern_error"] + NOISE_WEIGHT * (0.0 if np.isnan(noise) else noise)
    )
    return scores


def drift_score(yearly_ke: list[float], truth_yearly_ke: list[float], window: int = 5) -> dict[str, float]:
    """Long-run drift: share of years outside the truth's yearly KE range and the late/early KE log-ratio."""

    ke = np.asarray(yearly_ke, dtype=float)
    low, high = min(truth_yearly_ke), max(truth_yearly_ke)
    outside = float(np.mean((ke < low) | (ke > high)))
    window = max(1, min(window, len(ke) // 2))
    trend = abs(float(np.log(ke[-window:].mean() / ke[:window].mean())))
    return {"years_outside_truth_range": outside, "ke_trend": trend, "drift_score": outside + trend}


def summarise(rollouts: list[dict[str, Any]], first_epoch: int = 30, last_epoch: int = 60) -> dict[str, float]:
    """Average each scorecard entry over the scheduled rollouts in ``[first_epoch, last_epoch]``."""

    chosen = [r["scorecard"] for r in rollouts if first_epoch <= r["epoch"] <= last_epoch and "scorecard" in r]
    if not chosen:
        return {}
    return {key: float(np.mean([c[key] for c in chosen])) for key in chosen[0]}
