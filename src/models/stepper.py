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

    ``forcing_window`` is how many consecutive wind snapshots (from the step's
    start time onwards) the network sees; ``input_mean``/``input_std`` then
    carry one entry per forcing snapshot after the state channels.

    With ``history_frames=2`` the network also sees the latest tendency
    (state minus previous state), normalised by ``tendency_mean``/``tendency_std``
    and appended after the forcing channels, so it can infer how eddies and
    fronts are moving rather than only where they are.

    With ``mass_conserving`` the per-layer domain-mean thickness change is
    projected out (over wet cells when a fixed mask is given), so each layer's volume is conserved exactly (as in the
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
        static: np.ndarray | None = None,
        fixed_mask: np.ndarray | None = None,
        fixed_values: np.ndarray | None = None,
        forcing_window: int = 1,
        history_frames: int = 1,
        tendency_mean: np.ndarray | None = None,
        tendency_std: np.ndarray | None = None,
    ):
        super().__init__()
        if target_mode not in {"next_state", "residual"}:
            raise ValueError(f"unknown target_mode {target_mode!r}")
        if forcing_window < 1:
            raise ValueError("forcing_window must be at least 1")
        self.net = net
        self.forcing_window = forcing_window
        if history_frames not in {1, 2}:
            raise ValueError("history_frames must be 1 or 2")
        if history_frames == 2 and (tendency_mean is None or tendency_std is None):
            raise ValueError("history_frames=2 needs tendency_mean and tendency_std")
        self.history_frames = history_frames
        tendency = (np.zeros(0), np.zeros(0)) if history_frames == 1 else (tendency_mean, tendency_std)
        self.register_buffer("tendency_mean", torch.as_tensor(tendency[0], dtype=torch.float32).reshape(1, -1, 1, 1))
        self.register_buffer("tendency_std", torch.as_tensor(tendency[1], dtype=torch.float32).reshape(1, -1, 1, 1))
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

    def forward(self, state: torch.Tensor, forcing: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
        if forcing.dim() == 3:
            forcing = forcing[:, None]
        x = (torch.cat([state, forcing], dim=1) - self.input_mean) / self.input_std
        if self.history_frames == 2:
            if previous is None:
                raise ValueError("this stepper needs the previous state")
            x = torch.cat([x, (state - previous - self.tendency_mean) / self.tendency_std], dim=1)
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
        nxt = state + delta
        if has_fixed:
            nxt = torch.where(self.fixed_mask, self.fixed_values, nxt)
        return nxt


class StepperEmulator:
    """Adapts a ``StateStepper`` to the fixed evaluator's ``rollout`` interface."""

    def __init__(self, stepper: StateStepper, device: torch.device):
        self.stepper = stepper
        self.device = device

    def rollout(self, history: np.ndarray, forcing: np.ndarray, steps: int) -> np.ndarray:
        self.stepper.eval()
        start = len(history) - 1
        window = self.stepper.forcing_window
        current = torch.from_numpy(history[-1:].astype("float32")).to(self.device)
        previous = torch.from_numpy(history[-2:-1].astype("float32")).to(self.device) if self.stepper.history_frames == 2 else None
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for offset in range(steps):
                wind = torch.from_numpy(forcing[start + offset : start + offset + window][None].astype("float32")).to(self.device)
                nxt = self.stepper(current, wind, previous)
                previous = None if previous is None else current
                current = nxt
                predictions.append(current.cpu().numpy()[0])
        return np.stack(predictions)
