from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
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

    ``nudge_rate`` > 0 relaxes the large-scale part of the predicted anomaly
    from ``climate`` (the training-mean state) by that fraction per step. The
    large scale is a block average over ``nudge_pool``-cell blocks, bilinearly
    upsampled, so eddy-scale structure is left untouched; it is applied before
    land pinning and the volume projection, which therefore still hold.
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
        nudge_rate: float = 0.0,
        nudge_pool: int = 20,
        climate: np.ndarray | None = None,
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
        static_tensor = torch.zeros(0) if static is None else torch.as_tensor(static, dtype=torch.float32)[None]
        self.register_buffer("static", static_tensor)
        if (fixed_mask is None) != (fixed_values is None):
            raise ValueError("fixed_mask and fixed_values must be given together")
        mask = torch.zeros(0, dtype=torch.bool) if fixed_mask is None else torch.as_tensor(fixed_mask, dtype=torch.bool)[None, None]
        values = torch.zeros(0) if fixed_values is None else torch.as_tensor(fixed_values, dtype=torch.float32)[None]
        if nudge_rate < 0 or (nudge_rate > 0 and (climate is None or nudge_pool < 1)):
            raise ValueError("nudging needs nudge_rate >= 0, nudge_pool >= 1 and a climate state")
        self.nudge_rate = float(nudge_rate)
        self.nudge_pool = int(nudge_pool)
        self.register_buffer("climate", torch.zeros(0) if climate is None else torch.as_tensor(climate, dtype=torch.float32)[None])
        self.register_buffer("fixed_mask", mask)
        self.register_buffer("fixed_values", values)

    def forward(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        x = (torch.cat([state, forcing[:, None]], dim=1) - self.input_mean) / self.input_std
        if self.static.numel():
            x = torch.cat([x, self.static.expand(x.shape[0], -1, -1, -1)], dim=1)
        out = self.net(x) * self.output_std + self.output_mean
        if self.target_mode == "residual":
            delta = out
        else:
            delta = out - state
        if self.nudge_rate > 0:
            delta = delta - self.nudge_rate * self.large_scale(state + delta - self.climate)
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
        nxt = state + delta
        if has_fixed:
            nxt = torch.where(self.fixed_mask, self.fixed_values, nxt)
        return nxt


    def large_scale(self, field: torch.Tensor) -> torch.Tensor:
        ny, nx = field.shape[-2:]
        coarse = F.adaptive_avg_pool2d(field, (max(1, ny // self.nudge_pool), max(1, nx // self.nudge_pool)))
        return F.interpolate(coarse, size=(ny, nx), mode="bilinear", align_corners=False)


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
