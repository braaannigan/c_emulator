"""Latent reduced-order emulator: encoder, latent stepper, decoder.

The state is encoded once onto a coarse latent grid, stepped forward there
for the whole rollout, and decoded to physical space at every step. Errors
therefore evolve in a compressed space instead of being re-injected at grid
scale each step. Decoded fields keep the physical constraints: land cells are
pinned and each layer's wet-cell volume is restored to the start state's.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.evaluation import N_LAYERS
from src.models.unet import ConvBlock, StateUNet


class LatentModel(nn.Module):
    def __init__(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        forcing_mean: float,
        forcing_std: float,
        output_std: np.ndarray,
        static: np.ndarray | None = None,
        fixed_mask: np.ndarray | None = None,
        fixed_values: np.ndarray | None = None,
        latent_channels: int = 32,
        hidden_channels: int = 32,
        levels: int = 3,
        stepper_channels: int = 64,
        stepper_depth: int = 2,
        latent_noise_std: float = 0.0,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be at least 1")
        if (fixed_mask is None) != (fixed_values is None):
            raise ValueError("fixed_mask and fixed_values must be given together")
        n_state = len(state_mean)
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("output_std", torch.as_tensor(output_std, dtype=torch.float32).view(1, -1, 1, 1))
        self.forcing_mean, self.forcing_std = float(forcing_mean), float(forcing_std)
        self.register_buffer("static", torch.zeros(0) if static is None else torch.as_tensor(static, dtype=torch.float32)[None])
        self.register_buffer("fixed_mask", torch.zeros(0, dtype=torch.bool) if fixed_mask is None else torch.as_tensor(fixed_mask, dtype=torch.bool)[None, None])
        self.register_buffer("fixed_values", torch.zeros(0) if fixed_values is None else torch.as_tensor(fixed_values, dtype=torch.float32)[None])
        n_static = 0 if static is None else static.shape[0]
        self.levels = levels
        self.latent_noise_std = float(latent_noise_std)
        widths = [hidden_channels * 2**level for level in range(levels)]
        self.encoders = nn.ModuleList()
        previous = n_state + n_static
        for width in widths:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.to_latent = nn.Conv2d(widths[-1], latent_channels, 1)
        self.from_latent = nn.Conv2d(latent_channels, widths[-1], 1)
        self.decoders = nn.ModuleList(ConvBlock(widths[level], widths[max(level - 1, 0)]) for level in reversed(range(levels)))
        self.head = nn.Conv2d(widths[0] + n_static, n_state, 3, padding=1)
        self.stepper = StateUNet(latent_channels + 1 + n_static, latent_channels, stepper_channels, stepper_depth)
        # Bookkeeping expected by the training pipeline.
        self.target_mode = "latent"
        self.mass_conserving = True
        self.stochastic_channels = 0

    def _static_at(self, size: tuple[int, int], batch: int) -> torch.Tensor | None:
        if not self.static.numel():
            return None
        return F.interpolate(self.static, size=size, mode="bilinear", align_corners=False).expand(batch, -1, -1, -1)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        x = (state - self.state_mean) / self.state_std
        if self.static.numel():
            x = torch.cat([x, self.static.expand(x.shape[0], -1, -1, -1)], dim=1)
        for level, encoder in enumerate(self.encoders):
            if level:
                x = F.avg_pool2d(x, 2)
            x = encoder(x)
        return self.to_latent(F.avg_pool2d(x, 2))

    def step(self, latent: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        # Wind and static fields are smooth, so bilinear resizing to the latent grid suffices.
        wind = F.interpolate(((forcing - self.forcing_mean) / self.forcing_std)[:, None], size=latent.shape[-2:], mode="bilinear", align_corners=False)
        parts = [latent, wind]
        static = self._static_at(latent.shape[-2:], latent.shape[0])
        if static is not None:
            parts.append(static)
        return latent + self.stepper(torch.cat(parts, dim=1))

    def decode(self, latent: torch.Tensor, shape: tuple[int, int], volume: torch.Tensor) -> torch.Tensor:
        """Decode to physical units; ``volume`` is the per-layer wet-cell mean thickness to restore."""

        x = self.from_latent(latent)
        sizes = [(int(np.ceil(shape[0] / 2**level)), int(np.ceil(shape[1] / 2**level))) for level in range(self.levels)]
        for decoder, size in zip(self.decoders, reversed(sizes)):
            x = decoder(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        static = self._static_at(shape, x.shape[0])
        x = self.head(x if static is None else torch.cat([x, static], dim=1))
        state = x * self.state_std + self.state_mean
        if self.fixed_mask.numel():
            wet = (~self.fixed_mask).to(state.dtype)
            h = state[:, :N_LAYERS]
            h = h - ((h * wet).sum(dim=(2, 3), keepdim=True) / wet.sum() - volume) * wet
            state = torch.cat([h, state[:, N_LAYERS:]], dim=1)
            state = torch.where(self.fixed_mask, self.fixed_values, state)
        else:
            h = state[:, :N_LAYERS]
            state = torch.cat([h - h.mean(dim=(2, 3), keepdim=True) + volume, state[:, N_LAYERS:]], dim=1)
        return state

    def wet_volume(self, state: torch.Tensor) -> torch.Tensor:
        h = state[:, :N_LAYERS]
        if self.fixed_mask.numel():
            wet = (~self.fixed_mask).to(h.dtype)
            return (h * wet).sum(dim=(2, 3), keepdim=True) / wet.sum()
        return h.mean(dim=(2, 3), keepdim=True)

    def window_loss(self, states: torch.Tensor, winds: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """Reconstruction of the start state plus decoded unrolled-rollout MSE (output-std units).

        With ``latent_noise_std`` the latent start state is perturbed before the rollout.
        """

        start = states[:, 0]
        volume = self.wet_volume(start)
        if noise_std > 0:
            noise = torch.randn_like(start) * self.output_std * noise_std
            if self.fixed_mask.numel():
                noise = noise.masked_fill(self.fixed_mask, 0.0)
            start = start + noise
        shape = tuple(states.shape[-2:])
        latent = self.encode(start)
        loss = (((self.decode(latent, shape, volume) - states[:, 0]) / self.output_std) ** 2).mean()
        if self.latent_noise_std > 0:
            # Perturb the latent state itself (scaled by its spread), so the
            # latent stepper learns to damp errors in its own coordinates.
            latent = latent + torch.randn_like(latent) * latent.detach().std() * self.latent_noise_std
        steps = winds.shape[1]
        for k in range(steps):
            latent = self.step(latent, winds[:, k])
            loss = loss + (((self.decode(latent, shape, volume) - states[:, k + 1]) / self.output_std) ** 2).mean()
        return loss / (steps + 1)


class LatentEmulator:
    """Adapts a ``LatentModel`` to the fixed evaluator's ``rollout`` interface."""

    def __init__(self, model: LatentModel, device: torch.device):
        self.model = model
        self.device = device

    def rollout(self, history: np.ndarray, forcing: np.ndarray, steps: int) -> np.ndarray:
        self.model.eval()
        start = len(history) - 1
        state = torch.from_numpy(history[-1:].astype("float32")).to(self.device)
        shape = tuple(state.shape[-2:])
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            volume = self.model.wet_volume(state)
            latent = self.model.encode(state)
            for offset in range(steps):
                wind = torch.from_numpy(forcing[start + offset][None].astype("float32")).to(self.device)
                latent = self.model.step(latent, wind)
                predictions.append(self.model.decode(latent, shape, volume).cpu().numpy()[0])
        return np.stack(predictions)
