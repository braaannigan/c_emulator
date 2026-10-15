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

    ``static`` holds already-normalised time-invariant context channels
    ``(channels, y, x)`` appended to every network input.

    ``fixed_mask`` ``(y, x)`` marks land cells whose state never changes; they
    are pinned to ``fixed_values`` ``(channels, y, x)`` after every step.

    With ``mass_conserving`` the per-layer domain-mean thickness change is
    projected out (over wet cells when a fixed mask is given), so each layer's volume is conserved exactly (as in the
    source model).

    ``integrator`` ``euler`` adds one network increment per step; ``heun``
    treats the network as a tendency and applies a predictor-corrector step
    (two network evaluations), as in PEC-integrated ocean emulators.
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
        static: np.ndarray | None = None,
        fixed_mask: np.ndarray | None = None,
        fixed_values: np.ndarray | None = None,
        integrator: str = "euler",
    ):
        super().__init__()
        if target_mode not in {"next_state", "residual"}:
            raise ValueError(f"unknown target_mode {target_mode!r}")
        if integrator not in {"euler", "heun"}:
            raise ValueError(f"unknown integrator {integrator!r}")
        self.integrator = integrator
        self.net = net
        self.target_mode = target_mode
        self.mass_conserving = mass_conserving
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("input_std", torch.as_tensor(input_std, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("output_mean", torch.as_tensor(output_mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("output_std", torch.as_tensor(output_std, dtype=torch.float32)[None, :, None, None])
        static_tensor = torch.zeros(0) if static is None else torch.as_tensor(static, dtype=torch.float32)[None]
        self.register_buffer("static", static_tensor)
        if (fixed_mask is None) != (fixed_values is None):
            raise ValueError("fixed_mask and fixed_values must be given together")
        mask = torch.zeros(0, dtype=torch.bool) if fixed_mask is None else torch.as_tensor(fixed_mask, dtype=torch.bool)[None, None]
        values = torch.zeros(0) if fixed_values is None else torch.as_tensor(fixed_values, dtype=torch.float32)[None]
        self.register_buffer("fixed_mask", mask)
        self.register_buffer("fixed_values", values)

    def forward(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        delta = self.increment(state, forcing)
        if self.integrator == "heun":
            # Predictor-corrector (Heun): average the increment at the start
            # state and at the predicted state. The constraints are linear, so
            # the averaged increment still conserves volume and spares land.
            delta = 0.5 * (delta + self.increment(self.pin(state + delta), forcing))
        return self.pin(state + delta)

    def increment(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        """Network increment with land cells zeroed and (optionally) layer volume projected out."""

        x = (torch.cat([state, forcing[:, None]], dim=1) - self.input_mean) / self.input_std
        if self.static.numel():
            x = torch.cat([x, self.static.expand(x.shape[0], -1, -1, -1)], dim=1)
        out = self.net(x) * self.output_std + self.output_mean
        if self.target_mode == "residual":
            delta = out
        else:
            delta = out - state
        has_fixed = bool(self.fixed_mask.numel())
        if has_fixed:
            delta = delta.masked_fill(self.fixed_mask, 0.0)
        if self.mass_conserving:
            h_delta = delta[:, :N_LAYERS]
            if has_fixed:
                wet = (~self.fixed_mask).to(h_delta.dtype)
                h_delta = (h_delta - (h_delta * wet).sum(dim=(2, 3), keepdim=True) / wet.sum()) * wet
            else:
                h_delta = h_delta - h_delta.mean(dim=(2, 3), keepdim=True)
            delta = torch.cat([h_delta, delta[:, N_LAYERS:]], dim=1)
        return delta

    def pin(self, state: torch.Tensor) -> torch.Tensor:
        if self.fixed_mask.numel():
            return torch.where(self.fixed_mask, self.fixed_values, state)
        return state


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
