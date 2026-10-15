from __future__ import annotations

import numpy as np


def mse_by_time(truth: np.ndarray, rollout: np.ndarray) -> np.ndarray:
    """Mean squared error for each rollout time step."""

    if truth.shape != rollout.shape:
        raise ValueError(f"truth and rollout shapes differ: {truth.shape} != {rollout.shape}")
    return np.mean((truth - rollout) ** 2, axis=tuple(range(1, truth.ndim)))


def rmse_by_time(truth: np.ndarray, rollout: np.ndarray) -> np.ndarray:
    """Root mean squared error for each rollout time step."""

    return np.sqrt(mse_by_time(truth, rollout))


def summarise_mse(truth: np.ndarray, rollout: np.ndarray) -> tuple[float, float]:
    """Return mean MSE across rollout and MSE at the final rollout step."""

    series = mse_by_time(truth, rollout)
    return float(series.mean()), float(series[-1])
