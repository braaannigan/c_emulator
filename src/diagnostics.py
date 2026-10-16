"""Statistical-fidelity diagnostics for dgsw2l rollouts.

Beyond ~6-12 months a chaotic free run cannot follow the true eddies, so a
faithful emulator is judged on whether its *statistics* match the ocean
model's: wavenumber spectra (eddy scales), maps of eddy kinetic energy and
thickness variance (where the turbulence lives), decorrelation times and
frequency spectra (how fast eddies evolve), the response to the shifting
wind, and freedom from drift over long runs.

All functions take numpy arrays ``(time, y, x)`` (or state stacks
``(time, 6, y, x)`` ordered ``[h0, h1, u0, u1, v0, v1]``) restricted to wet
cells by the caller where it matters. They are diagnostics only and never
feed anything back into an emulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import xarray as xr

from src.data import analytic_surface_wind_stress
from src.evaluation import N_LAYERS

#: Wet interior of the dgsw2l grid (rows 0, 1, 198, 199 and columns 0, 99 contain land cells).
INTERIOR = (slice(2, -2), slice(1, -1))

#: Wavenumber bands in cycles per cell: basin/gyre scale, mesoscale eddies, near-grid scale.
SPATIAL_BANDS = {"large": (0.0, 0.05), "eddy": (0.05, 0.2), "small": (0.2, 0.5)}


def extended_forcing(ds: xr.Dataset, extra_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Analytic wind stress for the record plus ``extra_steps`` further output intervals.

    The wind is a known function of time (``src/data.py``), so long free runs
    beyond the end of the record can be forced exactly as the ocean model was.
    Returns ``(times_days, forcing)`` with forcing shaped ``(time, y, x)``.
    """

    times = ds["time_days"].values.astype("float64")
    interval = float(ds.attrs.get("output_interval_days", times[1] - times[0]))
    extra = times[-1] + interval * np.arange(1, extra_steps + 1)
    all_times = np.concatenate([times, extra])
    grid = xr.Dataset(coords={"time_days": all_times, "y": ds["y"].values, "x": ds["x"].values}, attrs=ds.attrs)
    return all_times, analytic_surface_wind_stress(grid)


def anomalies(field: np.ndarray) -> np.ndarray:
    return field - field.mean(axis=0, keepdims=True)


def _hann(n: int) -> np.ndarray:
    return np.hanning(n) if n > 1 else np.ones(1)


def wavenumber_spectrum(field: np.ndarray, axis: str = "x") -> tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum of time anomalies along ``x`` or ``y`` (Hann-windowed, cycles per cell)."""

    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    data = anomalies(np.asarray(field, dtype=np.float64))
    if axis == "y":
        data = np.swapaxes(data, 1, 2)
    n = data.shape[-1]
    data = data - data.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(data * _hann(n), axis=-1)) ** 2
    return np.fft.rfftfreq(n), power.mean(axis=(0, 1))


def frequency_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean temporal spectrum of each cell's anomaly series (cycles per step).

    Scaled so the non-zero frequencies sum to the mean anomaly variance, so
    band totals are variances and comparable between series of different length.
    """

    data = anomalies(np.asarray(field, dtype=np.float64))
    n = data.shape[0]
    power = (np.abs(np.fft.rfft(data * _hann(n)[:, None, None], axis=0)) ** 2).mean(axis=(1, 2))
    total = power[1:].sum()
    scale = (data**2).mean() / total if total > 0 else 0.0
    return np.fft.rfftfreq(n), power * scale


def band_power(freq: np.ndarray, power: np.ndarray, bands: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Total power in each half-open band ``[low, high)``; zero frequency excluded."""

    return {name: float(power[(freq > 0) & (freq >= low) & (freq < high)].sum()) for name, (low, high) in bands.items()}


def eke_map(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Eddy kinetic energy per unit mass, 0.5 * (var_t(u) + var_t(v))."""

    return 0.5 * (np.var(u, axis=0) + np.var(v, axis=0))


def variance_map(field: np.ndarray) -> np.ndarray:
    return np.var(field, axis=0)


def pattern_stats(model_map: np.ndarray, truth_map: np.ndarray) -> dict[str, float]:
    """Domain-mean ratio (model / truth) and spatial pattern correlation of two maps."""

    a, b = np.ravel(model_map), np.ravel(truth_map)
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"ratio": float(a.mean() / b.mean()), "pattern_correlation": correlation}


def autocorrelation(field: np.ndarray, max_lag: int) -> np.ndarray:
    """Domain-mean lag correlation of each cell's anomaly series for lags 0..max_lag."""

    data = anomalies(np.asarray(field, dtype=np.float64))
    n = data.shape[0]
    if not 0 < max_lag < n:
        raise ValueError("max_lag must be between 1 and len(series) - 1")
    variance = (data**2).mean(axis=0)
    wet = variance > 0
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        covariance = (data[lag:] * data[:-lag]).mean(axis=0)
        acf.append(float((covariance[wet] / variance[wet]).mean()))
    return np.array(acf)


def efolding_lag(acf: np.ndarray) -> float:
    """First lag (linearly interpolated) where the autocorrelation falls below 1/e; inf if never."""

    threshold = np.exp(-1.0)
    below = np.nonzero(acf < threshold)[0]
    if len(below) == 0:
        return float("inf")
    lag = int(below[0])
    upper, lower = acf[lag - 1], acf[lag]
    return float(lag - 1 + (upper - threshold) / (upper - lower))


def wind_response_r2(field: np.ndarray, times_days: np.ndarray, period_days: float) -> float:
    """Domain-mean R^2 of each cell's anomaly regressed on sin/cos of the wind-shift phase."""

    data = anomalies(np.asarray(field, dtype=np.float64)).reshape(len(times_days), -1)
    phase = 2.0 * np.pi * np.asarray(times_days, dtype=np.float64) / period_days
    design = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(phase)])
    coefficients, *_ = np.linalg.lstsq(design, data, rcond=None)
    residual = data - design @ coefficients
    total = (data**2).sum(axis=0)
    wet = total > 0
    return float((1.0 - (residual**2).sum(axis=0)[wet] / total[wet]).mean())


def kinetic_energy_series(states: np.ndarray) -> np.ndarray:
    """Domain-mean kinetic energy per unit mass for each step and layer, shape ``(time, layers)``."""

    u, v = states[:, N_LAYERS : 2 * N_LAYERS], states[:, 2 * N_LAYERS : 3 * N_LAYERS]
    return 0.5 * (u**2 + v**2).mean(axis=(2, 3))


def relative_vorticity(u: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Relative vorticity dv/dx - du/dy of cell-centred velocities ``(..., y, x)`` by centred differences."""

    return np.gradient(v, dx, axis=-1) - np.gradient(u, dx, axis=-2)


def block_statistics(states: np.ndarray, block: int) -> dict[str, list[float]]:
    """Per-block (e.g. per-year) domain-mean KE and thickness std, to expose slow drift."""

    if block < 1:
        raise ValueError("block must be positive")
    n_blocks = len(states) // block
    ke, h_std = [], []
    for index in range(n_blocks):
        chunk = states[index * block : (index + 1) * block]
        ke.append(float(kinetic_energy_series(chunk).mean()))
        h_std.append(float(np.sqrt(np.var(chunk[:, :N_LAYERS], axis=0).mean())))
    return {"kinetic_energy": ke, "thickness_std": h_std}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def compare_statistics(model: np.ndarray, truth: np.ndarray, times_model: np.ndarray, times_truth: np.ndarray, wind_period_days: float, max_lag: int = 20) -> dict[str, Any]:
    """Fidelity report comparing a model state stack with a truth state stack (interior cells).

    The two stacks may cover different periods and lengths (e.g. a long free
    run against the training record); only their statistics are compared.
    """

    ys, xs = INTERIOR
    model, truth = model[:, :, ys, xs], truth[:, :, ys, xs]
    report: dict[str, Any] = {"layers": {}}
    for layer in range(N_LAYERS):
        h, u, v = layer, N_LAYERS + layer, 2 * N_LAYERS + layer
        entry: dict[str, Any] = {}
        for name, index in (("thickness", h), ("zonal_velocity", u)):
            for axis in ("x", "y"):
                freq, pm = wavenumber_spectrum(model[:, index], axis)
                _, pt = wavenumber_spectrum(truth[:, index], axis)
                bm, bt = band_power(freq, pm, SPATIAL_BANDS), band_power(freq, pt, SPATIAL_BANDS)
                entry[f"{name}_spectrum_{axis}_band_ratio"] = {band: _ratio(bm[band], bt[band]) for band in SPATIAL_BANDS}
        entry["eke_map"] = pattern_stats(eke_map(model[:, u], model[:, v]), eke_map(truth[:, u], truth[:, v]))
        entry["thickness_variance_map"] = pattern_stats(variance_map(model[:, h]), variance_map(truth[:, h]))
        lag = min(max_lag, len(model) - 1, len(truth) - 1)
        acf_model, acf_truth = autocorrelation(model[:, h], lag), autocorrelation(truth[:, h], lag)
        entry["thickness_efolding_steps"] = {"model": efolding_lag(acf_model), "truth": efolding_lag(acf_truth)}
        entry["thickness_autocorrelation"] = {"model": acf_model.tolist(), "truth": acf_truth.tolist()}
        fm, sm = frequency_spectrum(model[:, h])
        ft, st = frequency_spectrum(truth[:, h])
        temporal = {"slow": (0.0, 0.02), "seasonal": (0.02, 0.1), "fast": (0.1, 0.51)}
        bm, bt = band_power(fm, sm, temporal), band_power(ft, st, temporal)
        entry["thickness_frequency_band_ratio"] = {band: _ratio(bm[band], bt[band]) for band in temporal}
        entry["wind_response_r2"] = {
            "model": wind_response_r2(model[:, h], times_model, wind_period_days),
            "truth": wind_response_r2(truth[:, h], times_truth, wind_period_days),
        }
        report["layers"][str(layer)] = entry
    report["kinetic_energy_ratio"] = float(kinetic_energy_series(model).mean() / kinetic_energy_series(truth).mean())
    report["thickness_std_ratio"] = float(np.sqrt(np.var(model[:, :N_LAYERS], axis=0).mean() / np.var(truth[:, :N_LAYERS], axis=0).mean()))
    return report
