# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Spatial Softmax layer for extracting spatial features from feature maps.

    Given feature maps of shape (B, C, H, W), computes a spatial soft-argmax to
    produce (x, y) coordinates for each channel, resulting in output (B, C*2).
    """

    def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature

        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij"
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = features.shape
        features_flat = features.view(b, c, -1)  # (B, C, H*W)
        attention = F.softmax(features_flat / self.temperature, dim=-1)  # (B, C, H*W)
        x_exp = torch.sum(self.pos_x * attention, dim=-1, keepdim=True)  # (B, C, 1)
        y_exp = torch.sum(self.pos_y * attention, dim=-1, keepdim=True)  # (B, C, 1)
        coords = torch.cat([x_exp, y_exp], dim=-1)  # (B, C, 2)
        return coords.view(b, -1)  # (B, C*2)
