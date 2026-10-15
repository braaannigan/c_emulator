from __future__ import annotations

import torch
from torch import nn


class StateCNN(nn.Module):
    """Plain convolutional next-step predictor for the two-layer state."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        hidden_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        previous_channels = in_channels
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Conv2d(previous_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                    nn.GELU(),
                ]
            )
            previous_channels = hidden_channels
        layers.append(nn.Conv2d(previous_channels, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
