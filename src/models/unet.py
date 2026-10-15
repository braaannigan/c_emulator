from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "zeros"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateUNet(nn.Module):
    """Multi-scale U-Net next-step network.

    Each level halves the resolution (2x2 average pooling), so ``depth`` levels
    widen the receptive field by roughly ``2**depth`` -- enough to follow
    eddies that move tens of cells between 7-day snapshots. Decoder features
    are resized to the skip connection's shape, so odd grid sizes work.
    """

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32, depth: int = 3, padding_mode: str = "zeros"):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        widths = [base_channels * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(ConvBlock(previous, width, padding_mode))
            previous = width
        self.decoders = nn.ModuleList(ConvBlock(widths[level + 1] + widths[level], widths[level], padding_mode) for level in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            if level:
                x = F.avg_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        x = skips.pop()
        for decoder in self.decoders:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)
