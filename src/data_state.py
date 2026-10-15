from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from src.data import analytic_surface_wind_stress, centre_meridional_velocity, centre_zonal_velocity, chronological_pair_split


@dataclass(frozen=True)
class ChannelNormalisation:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


def load_two_layer_state_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, xr.Dataset]:
    """Load two-layer state inputs and targets from a double-gyre NetCDF file.

    State channels are ordered as all layer thicknesses, all centred zonal
    velocities, then all centred meridional velocities. A reconstructed wind
    stress channel is appended to inputs but not predicted.
    """

    ds = xr.open_dataset(path)
    h = ds["layer_thickness"].values.astype("float32")
    u = centre_zonal_velocity(ds["zonal_velocity"].values.astype("float32"))
    v = centre_meridional_velocity(ds["meridional_velocity"].values.astype("float32"))
    if h.shape[1] != 2:
        raise ValueError(f"expected exactly two layers, found {h.shape[1]}")
    state = np.concatenate([h, u, v], axis=1).astype("float32")
    wind = analytic_surface_wind_stress(ds)[:, None]
    inputs = np.concatenate([state, wind], axis=1).astype("float32")
    return inputs, state, ds


def fit_channel_normalisation(inputs: np.ndarray, targets: np.ndarray, indices: np.ndarray) -> ChannelNormalisation:
    x = inputs[indices]
    y = targets[indices + 1]
    input_mean = x.mean(axis=(0, 2, 3), dtype=np.float64).astype("float32")
    input_std = x.std(axis=(0, 2, 3), dtype=np.float64).astype("float32")
    target_mean = y.mean(axis=(0, 2, 3), dtype=np.float64).astype("float32")
    target_std = y.std(axis=(0, 2, 3), dtype=np.float64).astype("float32")
    input_std = np.where(input_std < 1e-6, 1.0, input_std).astype("float32")
    target_std = np.where(target_std < 1e-6, 1.0, target_std).astype("float32")
    return ChannelNormalisation(input_mean, input_std, target_mean, target_std)


class NextStepStateDataset(Dataset):
    """Next-step dataset for two-layer h/u/v state prediction."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
        norm: ChannelNormalisation,
    ):
        self.inputs = inputs
        self.targets = targets
        self.indices = indices.astype("int64")
        self.norm = norm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[item]
        x = (self.inputs[t] - self.norm.input_mean[:, None, None]) / self.norm.input_std[:, None, None]
        y = (self.targets[t + 1] - self.norm.target_mean[:, None, None]) / self.norm.target_std[:, None, None]
        return torch.from_numpy(x.astype("float32")), torch.from_numpy(y.astype("float32"))


def dgsw2l_pair_split(n_times: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    return chronological_pair_split(n_times, train_fraction)


def fit_increment_normalisation(state: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std of one-step increments ``state[t+1] - state[t]`` over ``indices``."""

    delta = state[indices + 1] - state[indices]
    mean = delta.mean(axis=(0, 2, 3), dtype=np.float64).astype("float32")
    std = delta.std(axis=(0, 2, 3), dtype=np.float64).astype("float32")
    return mean, np.where(std < 1e-6, 1.0, std).astype("float32")


class RolloutWindowDataset(Dataset):
    """Windows of ``unroll_steps + 1`` consecutive physical states plus the wind at each start.

    ``indices`` are next-step pair starts; only starts whose whole window stays
    inside those pairs are used, so no target leaves the training split.
    """

    def __init__(self, state: np.ndarray, forcing: np.ndarray, indices: np.ndarray, unroll_steps: int = 1):
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be at least 1")
        last_target = int(indices.max()) + 1
        self.starts = np.array([t for t in indices if t + unroll_steps <= last_target], dtype="int64")
        if len(self.starts) == 0:
            raise ValueError("no training window fits inside the training split")
        self.state = state
        self.forcing = forcing
        self.unroll_steps = unroll_steps

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.starts[item])
        k = self.unroll_steps
        return torch.from_numpy(self.state[t : t + k + 1].astype("float32")), torch.from_numpy(self.forcing[t : t + k].astype("float32"))
