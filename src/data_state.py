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
