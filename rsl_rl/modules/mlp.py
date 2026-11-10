# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import get_param, resolve_nn_activation


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
        """
        super().__init__()

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    @torch.jit.unused
    def output_layer(self) -> nn.Linear:
        """Return the output layer for distribution-specific initialization."""
        return [layer for layer in self if isinstance(layer, nn.Linear)][-1]

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x


class ResidualMLP(nn.Module):
    r"""Residual MLP with batch normalization and hyperspherical weight normalization.

    The network follows the architecture used by FlashSAC and SimbaV2:

    - The input passes through batch normalization and a bias-free linear embedding layer.
    - The embedding passes through a stack of inverted-bottleneck residual blocks, each containing two sequences of
      a linear layer, batch normalization, and ReLU, followed by a skip connection.
    - The final latent passes through RMS normalization and a linear output layer.

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
        batch_norm_momentum: float = 0.01,
    ) -> None:
        """Initialize the residual MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dim: Width of the residual stream.
            num_blocks: Number of residual blocks.
            expansion: Width multiplier of the inner layer of each residual block.
            batch_norm_momentum: Momentum of the batch normalization running statistics.
        """
        super().__init__()

        # Resolve the output shape and flattened output dimension
        self.output_shape = (output_dim,) if isinstance(output_dim, int) else tuple(output_dim)
        total_out_dim = reduce(lambda x, y: x * y, self.output_shape)

        # Create the input embedding, residual blocks, and output projection
        self.input_norm = nn.BatchNorm1d(input_dim, momentum=batch_norm_momentum)
        self.input_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([
            _ResidualBlock(hidden_dim, expansion, batch_norm_momentum) for _ in range(num_blocks)
        ])
        self.output_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.output_layer = nn.Linear(hidden_dim, total_out_dim)

        # Initialize linear weights and normalize the network parameters
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.normalize_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual MLP."""
        # Flatten leading dimensions for batch normalization
        batch_shape = x.shape[:-1]
        x = x.flatten(0, -2)

        # Apply the input layer and residual blocks
        x = self.input_layer(self.input_norm(x))
        for block in self.blocks:
            x = block(x)

        # Project the output and restore its leading dimensions
        if torch.jit.is_scripting():  # type: ignore
            x = self.output_norm(x)
        elif torch.onnx.is_in_onnx_export():  # Compute manually since ONNX does not support RMSNorm
            x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.output_norm.eps)  # type: ignore
            x = x * self.output_norm.weight
        else:
            x = self.output_norm(x)
        x = self.output_layer(x)
        return x.unflatten(-1, self.output_shape).unflatten(0, batch_shape)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Project weights onto the unit sphere and rescale normalization parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Normalize each row of the weight matrix
                module.weight.copy_(torch.nn.functional.normalize(module.weight, dim=-1, eps=1e-8))
            elif isinstance(module, nn.BatchNorm1d):
                # Normalize the scale and bias jointly to preserve their relative magnitudes
                squared_norm = module.weight.square().sum() + module.bias.square().sum()
                factor = math.sqrt(module.weight.shape[-1]) * torch.rsqrt(squared_norm + 1e-8)
                module.weight.mul_(factor)
                module.bias.mul_(factor)
            elif isinstance(module, nn.RMSNorm):
                # Rescale the RMS normalization weights to the square root of the feature dimension
                factor = math.sqrt(module.weight.shape[-1]) * torch.rsqrt(module.weight.square().sum() + 1e-8)
                module.weight.mul_(factor)


class _ResidualBlock(nn.Module):
    """Inverted-bottleneck residual block with batch normalization."""

    def __init__(self, hidden_dim: int, expansion: int, batch_norm_momentum: float) -> None:
        """Initialize the expanded linear layers and batch normalization."""
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * expansion, bias=False)
        self.norm1 = nn.BatchNorm1d(hidden_dim * expansion, momentum=batch_norm_momentum)
        self.linear2 = nn.Linear(hidden_dim * expansion, hidden_dim, bias=False)
        self.norm2 = nn.BatchNorm1d(hidden_dim, momentum=batch_norm_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block and add the input."""
        y = torch.nn.functional.relu(self.norm1(self.linear1(x)))
        y = torch.nn.functional.relu(self.norm2(self.linear2(y)))
        return x + y
