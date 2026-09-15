from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Residual2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.body(x)


def _residual_stack(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(Residual2D(channels, channels) for _ in range(count)))


class Hourglass(nn.Module):
    def __init__(self, depth: int, channels: int, blocks_per_stage: int = 2) -> None:
        super().__init__()
        self.upper = _residual_stack(channels, blocks_per_stage)
        self.lower_input = _residual_stack(channels, blocks_per_stage)
        self.inner = (
            Hourglass(depth - 1, channels, blocks_per_stage)
            if depth > 1
            else _residual_stack(channels, blocks_per_stage)
        )
        self.lower_output = _residual_stack(channels, blocks_per_stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upper = self.upper(x)
        lower = F.avg_pool2d(x, 2, stride=2)
        lower = self.lower_output(self.inner(self.lower_input(lower)))
        lower = F.interpolate(lower, size=upper.shape[-2:], mode="bilinear", align_corners=True)
        return upper + lower


class StackedHourglassEncoder(nn.Module):
    """Four-stack image encoder compatible with ICON-style pixel-aligned queries."""

    def __init__(
        self,
        input_channels: int = 3,
        width: int = 128,
        feature_dim: int = 64,
        num_stacks: int = 4,
        hourglass_depth: int = 3,
        blocks_per_stack: int = 2,
    ) -> None:
        super().__init__()
        self.num_stacks = int(num_stacks)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width // 2, 7, stride=2, padding=3),
            nn.GroupNorm(_groups(width // 2), width // 2),
            nn.SiLU(),
            Residual2D(width // 2, width),
            nn.AvgPool2d(2, stride=2),
            Residual2D(width, width),
        )
        self.hourglasses = nn.ModuleList(
            Hourglass(hourglass_depth, width, blocks_per_stack)
            for _ in range(self.num_stacks)
        )
        self.post = nn.ModuleList(
            nn.Sequential(
                _residual_stack(width, blocks_per_stack),
                nn.Conv2d(width, width, 1),
                nn.GroupNorm(_groups(width), width),
                nn.SiLU(),
            )
            for _ in range(self.num_stacks)
        )
        self.feature_heads = nn.ModuleList(
            nn.Conv2d(width, feature_dim, 1) for _ in range(self.num_stacks)
        )
        self.merge_hidden = nn.ModuleList(
            nn.Conv2d(width, width, 1) for _ in range(self.num_stacks - 1)
        )
        self.merge_features = nn.ModuleList(
            nn.Conv2d(feature_dim, width, 1) for _ in range(self.num_stacks - 1)
        )

    def forward(self, image: torch.Tensor, return_all: bool = False):
        hidden = self.stem(image)
        outputs: List[torch.Tensor] = []
        for index in range(self.num_stacks):
            post = self.post[index](self.hourglasses[index](hidden))
            features = self.feature_heads[index](post)
            outputs.append(features)
            if index < self.num_stacks - 1:
                hidden = hidden + self.merge_hidden[index](post) + self.merge_features[index](features)
        return outputs if return_all else outputs[-1]

