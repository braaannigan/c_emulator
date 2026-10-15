from __future__ import annotations

import numpy as np
import xarray as xr


def centre_zonal_velocity(u: np.ndarray) -> np.ndarray:
    """Centre C-grid zonal velocity from x faces onto thickness cell centres."""

    return 0.5 * (u[..., :-1] + u[..., 1:])


def centre_meridional_velocity(v: np.ndarray) -> np.ndarray:
    """Centre C-grid meridional velocity from y faces onto thickness cell centres."""

    return 0.5 * (v[..., :-1, :] + v[..., 1:, :])


def chronological_pair_split(n_times: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return next-step pair start indices for first-N train / last-N eval split."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if n_times < 3:
        raise ValueError("at least three time steps are required")
    split_time = int(n_times * train_fraction)
    split_time = min(max(split_time, 2), n_times - 1)
    train = np.arange(0, split_time - 1)
    eval_ = np.arange(split_time - 1, n_times - 1)
    return train, eval_


def analytic_surface_wind_stress(ds: xr.Dataset) -> np.ndarray:
    """Construct the scalar wind-stress channel when it is not stored in the dataset.

    The double-gyre dataset stores velocities and thickness plus metadata
    describing a sinusoidally shifting wind. This returns a deterministic
    zonal-stress proxy on the thickness grid with shape ``(time, y, x)``.
    """

    times = ds["time_days"].values.astype("float32")
    y = ds["y"].values.astype("float32")
    x = ds["x"].values.astype("float32")
    tau_max = float(ds.attrs.get("wind_stress_max", 0.0))
    shift_amp = float(ds.attrs.get("wind_shift_amplitude_m", 0.0))
    period = float(ds.attrs.get("wind_shift_period_days", 1.0))
    ly = float(y.max() - y.min() + (y[1] - y[0] if len(y) > 1 else 1.0))

    phase = 2.0 * np.pi * times / period
    shifts = shift_amp * np.sin(phase)
    shifted_y = y[None, :, None] - shifts[:, None, None]
    tau_y = tau_max * np.cos(2.0 * np.pi * shifted_y / ly)
    return np.broadcast_to(tau_y, (len(times), len(y), len(x))).astype("float32")
