from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.evaluation import N_LAYERS


class StateStepper(nn.Module):
    """Differentiable one-step map from a physical state (and wind) to the next state.

    Wraps a network that acts in normalised units. ``target_mode`` selects how
    the network output is interpreted:

    - ``next_state``: the output is the normalised next state;
    - ``residual``: the output is the normalised increment, added to the
      current state.

    With ``mass_conserving`` the per-layer domain-mean thickness change is
    projected out, so each layer's volume is conserved exactly (as in the
    source model).
    """

    def __init__(
        self,
        net: nn.Module,
        input_mean: np.ndarray,
        input_std: np.ndarray,
        output_mean: np.ndarray,
        output_std: np.ndarray,
        target_mode: str = "next_state",
        mass_conserving: bool = False,
    ):
        super().__init__()
        if target_mode not in {"next_state", "residual"}:
            raise ValueError(f"unknown target_mode {target_mode!r}")
        self.net = net
        self.target_mode = target_mode
        self.mass_conserving = mass_conserving
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("input_std", torch.as_tensor(input_std, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("output_mean", torch.as_tensor(output_mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("output_std", torch.as_tensor(output_std, dtype=torch.float32)[None, :, None, None])

    def forward(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, forcing[:, None]], dim=1)
        out = self.net((x - self.input_mean) / self.input_std) * self.output_std + self.output_mean
        if self.target_mode == "residual":
            delta = out
        else:
            delta = out - state
        if self.mass_conserving:
            h_delta = delta[:, :N_LAYERS]
            h_delta = h_delta - h_delta.mean(dim=(2, 3), keepdim=True)
            delta = torch.cat([h_delta, delta[:, N_LAYERS:]], dim=1)
        return state + delta


class StepperEmulator:
    """Adapts a ``StateStepper`` to the fixed evaluator's ``rollout`` interface."""

    def __init__(self, stepper: StateStepper, device: torch.device):
        self.stepper = stepper
        self.device = device

    def rollout(self, history: np.ndarray, forcing: np.ndarray, steps: int) -> np.ndarray:
        self.stepper.eval()
        start = len(history) - 1
        current = torch.from_numpy(history[-1:].astype("float32")).to(self.device)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for offset in range(steps):
                wind = torch.from_numpy(forcing[start + offset][None].astype("float32")).to(self.device)
                current = self.stepper(current, wind)
                predictions.append(current.cpu().numpy()[0])
        return np.stack(predictions)
