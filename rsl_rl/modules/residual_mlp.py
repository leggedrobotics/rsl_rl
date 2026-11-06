# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from functools import reduce


class ResidualMLP(nn.Module):
    r"""Residual MLP with batch normalization and hyperspherical weight normalization.

    The network follows the architecture used by FlashSAC and SimbaV2:

    - The input is normalized with batch normalization and embedded with a bias-free linear layer.
    - A stack of inverted-bottleneck residual blocks (linear, batch norm, ReLU, linear, batch norm, ReLU, skip).
    - RMS normalization of the final latent, followed by a linear output layer.

    Calling :meth:`normalize_parameters` after every optimizer step projects the rows of all linear weight matrices
    onto the unit sphere and rescales the normalization parameters to norm :math:`\sqrt{d}`. Since the input is
    normalized by batch normalization, no separate observation normalization is needed.

    References:
        - Kim et al. "FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control."
          arXiv preprint arXiv:2604.04539 (2026).
        - Lee et al. "Hyperspherical Normalization for Scalable Deep Reinforcement Learning." ICML (2025).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expansion: int = 4,
        bn_momentum: float = 0.01,
    ) -> None:
        """Initialize the residual MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output. If a tuple, the output is reshaped to the desired shape.
            hidden_dim: Width of the residual stream.
            num_blocks: Number of residual blocks.
            expansion: Width multiplier of the inner layer of each residual block.
            bn_momentum: Momentum of the batch normalization running statistics.
        """
        super().__init__()
        self.output_shape = (output_dim,) if isinstance(output_dim, int) else tuple(output_dim)
        total_out_dim = reduce(lambda x, y: x * y, self.output_shape)

        self.input_norm = nn.BatchNorm1d(input_dim, momentum=bn_momentum)
        self.embed = nn.Linear(input_dim, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden_dim, expansion, bn_momentum) for _ in range(num_blocks)])
        self.output_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.output = nn.Linear(hidden_dim, total_out_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.normalize_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual MLP."""
        batch_shape = x.shape[:-1]
        x = x.flatten(0, -2)
        x = self.embed(self.input_norm(x))
        for block in self.blocks:
            x = block(x)
        x = self.output(self.output_norm(x))
        return x.unflatten(-1, self.output_shape).unflatten(0, batch_shape)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Project weights onto the unit sphere and rescale normalization parameters to norm sqrt(d)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(F.normalize(module.weight, dim=-1, eps=1e-8))
            elif isinstance(module, nn.BatchNorm1d):
                sq_norm = module.weight.square().sum() + module.bias.square().sum()
                factor = math.sqrt(module.weight.shape[-1]) * torch.rsqrt(sq_norm + 1e-8)
                module.weight.mul_(factor)
                module.bias.mul_(factor)
            elif isinstance(module, nn.RMSNorm):
                factor = math.sqrt(module.weight.shape[-1]) * torch.rsqrt(module.weight.square().sum() + 1e-8)
                module.weight.mul_(factor)


class _ResidualBlock(nn.Module):
    """Inverted-bottleneck residual block with batch normalization."""

    def __init__(self, hidden_dim: int, expansion: int, bn_momentum: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * expansion, bias=False)
        self.norm1 = nn.BatchNorm1d(hidden_dim * expansion, momentum=bn_momentum)
        self.linear2 = nn.Linear(hidden_dim * expansion, hidden_dim, bias=False)
        self.norm2 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.linear1(x)))
        y = F.relu(self.norm2(self.linear2(y)))
        return x + y
