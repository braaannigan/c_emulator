"""Load a trained dgsw2l stepper emulator together with its data, for analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.config import ExperimentConfig
from src.data_state import dgsw2l_pair_split, load_two_layer_state_arrays
from src.evaluation import STATE_CHANNELS
from src.models.stepper import StepperEmulator
from src.pipelines.dgsw2l_baseline import build_stepper


@dataclass
class LoadedEmulator:
    emulator: StepperEmulator
    state: np.ndarray
    forcing: np.ndarray
    source: xr.Dataset
    train_indices: np.ndarray
    eval_indices: np.ndarray


def default_checkpoint(config: ExperimentConfig) -> Path:
    return config.results_csv.parent / "checkpoints" / f"{config.experiment_id}.pt"


def load_trained_emulator(config: ExperimentConfig, checkpoint: Path | None = None, device: torch.device | None = None) -> LoadedEmulator:
    """Rebuild the configured stepper, load its checkpoint and wrap it for rollouts."""

    device = device or torch.device("cpu")
    inputs, state, source = load_two_layer_state_arrays(config.training_data)
    train_indices, eval_indices = dgsw2l_pair_split(len(state), config.train_fraction)
    static_fields = [str(name) for name in config.raw.get("inputs", {}).get("static_fields", [])]
    stepper = build_stepper(config.raw.get("model", {}), inputs, state, train_indices, static_fields)
    stepper.load_state_dict(torch.load(checkpoint or default_checkpoint(config), map_location="cpu"))
    stepper.to(device)
    return LoadedEmulator(StepperEmulator(stepper, device), state, inputs[:, STATE_CHANNELS], source, train_indices, eval_indices)
