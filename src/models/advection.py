"""Semi-Lagrangian advection prior for the two-layer state.

Each layer's fields (h, u, v) are carried along that layer's own frozen
velocity field over one output interval: departure points are found by
integrating backwards in ``substeps`` sub-steps, and fields are sampled there
bilinearly. This gives the network an explicit guess of where water, eddies
and fronts move between snapshots, which a local stencil cannot see.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.evaluation import N_LAYERS


def advect_state(state: torch.Tensor, dt_seconds: float, dx_metres: float, substeps: int) -> torch.Tensor:
    """Advect ``state`` ``(batch, 6, y, x)`` ordered ``[h0, h1, u0, u1, v0, v1]``."""

    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    batch, _, ny, nx = state.shape
    ys = torch.linspace(-1 + 1 / ny, 1 - 1 / ny, ny, device=state.device, dtype=state.dtype)
    xs = torch.linspace(-1 + 1 / nx, 1 - 1 / nx, nx, device=state.device, dtype=state.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).expand(batch, ny, nx, 2)
    # metres -> normalised grid units (grid_sample spans [-1, 1] over nx / ny cells)
    scale = torch.tensor([2.0 / (nx * dx_metres), 2.0 / (ny * dx_metres)], device=state.device, dtype=state.dtype)
    step = dt_seconds / substeps
    layers = []
    for layer in range(N_LAYERS):
        fields = state[:, [layer, N_LAYERS + layer, 2 * N_LAYERS + layer]]
        velocity = fields[:, 1:]  # (batch, 2, y, x): u (x direction), v (y direction)
        points = base
        for _ in range(substeps):
            local = F.grid_sample(velocity, points, mode="bilinear", padding_mode="border", align_corners=False)
            points = points - step * local.permute(0, 2, 3, 1) * scale
        layers.append(F.grid_sample(fields, points, mode="bilinear", padding_mode="border", align_corners=False))
    h = torch.cat([part[:, :1] for part in layers], dim=1)
    u = torch.cat([part[:, 1:2] for part in layers], dim=1)
    v = torch.cat([part[:, 2:3] for part in layers], dim=1)
    return torch.cat([h, u, v], dim=1)
