"""Fixed rollout evaluation for dgsw2l emulators.

THIS FILE DEFINES THE BENCHMARK. Do not modify it as part of an experiment.

The evaluator owns the ground truth. An emulator only ever receives:

- ``history``: the true two-layer state up to and including the rollout start
  time (all of it is in the past, so it is fair to use);
- ``forcing``: the known surface wind-stress forcing from time 0 to the final
  rollout target time.

It must return its own autoregressive prediction of the full two-layer state
for every rollout step. No future truth is ever handed to the emulator, so
every predicted field (thickness *and* velocity) has to come from the model.

State channel order: ``[h_layer0, h_layer1, u_layer0, u_layer1, v_layer0, v_layer1]``
with velocities centred on the thickness grid.

Primary metric: ``eval_mse_mean`` = mean over rollout steps of the mean squared
error of the two layer-thickness channels (raw units, metres squared).
``eval_mse_last`` is the same quantity at the final rollout step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

N_LAYERS = 2
STATE_CHANNELS = 3 * N_LAYERS
THICKNESS = slice(0, N_LAYERS)
VELOCITY = slice(N_LAYERS, STATE_CHANNELS)


class Emulator(Protocol):
    def rollout(self, history: np.ndarray, forcing: np.ndarray, steps: int) -> np.ndarray:
        """Return predicted states with shape ``(steps, 6, y, x)``.

        ``history`` has shape ``(start + 1, 6, y, x)``; ``history[-1]`` is the
        state at the rollout start time. ``forcing`` has shape
        ``(start + 1 + steps, y, x)``; ``forcing[i]`` is the wind stress at time
        index ``i``. Prediction ``k`` is the state at time index ``start + 1 + k``.
        """
        ...


@dataclass(frozen=True)
class RolloutEvaluation:
    truth: np.ndarray
    prediction: np.ndarray
    thickness_mse_by_time: np.ndarray
    velocity_mse_by_time: np.ndarray

    @property
    def eval_mse_mean(self) -> float:
        return float(self.thickness_mse_by_time.mean())

    @property
    def eval_mse_last(self) -> float:
        return float(self.thickness_mse_by_time[-1])

    @property
    def velocity_mse_mean(self) -> float:
        return float(self.velocity_mse_by_time.mean())


def evaluate_rollout(
    emulator: Emulator,
    state: np.ndarray,
    forcing: np.ndarray,
    eval_indices: np.ndarray,
    max_steps: int | None = None,
) -> RolloutEvaluation:
    """Run a fully autoregressive rollout from the start of the eval split and score it."""

    if state.ndim != 4 or state.shape[1] != STATE_CHANNELS:
        raise ValueError(f"state must have shape (time, {STATE_CHANNELS}, y, x), got {state.shape}")
    if forcing.shape[0] != state.shape[0] or forcing.shape[1:] != state.shape[2:]:
        raise ValueError(f"forcing shape {forcing.shape} does not match state shape {state.shape}")
    steps = len(eval_indices) if max_steps is None else min(len(eval_indices), max_steps)
    if steps < 1:
        raise ValueError("rollout needs at least one step")
    start = int(eval_indices[0])
    history = state[: start + 1].copy()
    visible_forcing = forcing[: start + 1 + steps].copy()
    prediction = np.asarray(emulator.rollout(history, visible_forcing, steps), dtype="float32")
    truth = state[start + 1 : start + 1 + steps].astype("float32")
    if prediction.shape != truth.shape:
        raise ValueError(f"emulator returned shape {prediction.shape}, expected {truth.shape}")
    error = (prediction.astype("float64") - truth.astype("float64")) ** 2
    thickness_mse = error[:, THICKNESS].mean(axis=(1, 2, 3))
    velocity_mse = error[:, VELOCITY].mean(axis=(1, 2, 3))
    return RolloutEvaluation(truth, prediction, thickness_mse, velocity_mse)
