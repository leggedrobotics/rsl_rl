# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn as nn

from rsl_rl.utils import resolve_nn_activation


class CNN(nn.Sequential):
    """Convolutional Neural Network (CNN).

    The CNN network is a sequence of convolutional layers, optional batch normalization, activation functions, and
    optional max pooling. The final output can be flattened or pooled depending on the configuration.
    """

    def __init__(
        self,
        in_channels: int,
        activation: str,
        out_channels: list[int],
        kernel_size: list[tuple[int, int]] | tuple[int, int],
        stride: list[int] | int = 1,
        flatten: bool = True,
        avg_pool: tuple[int, int] | None = None,
        batchnorm: bool | list[bool] = False,
        max_pool: bool | list[bool] = False,
    ) -> None:
        """Initialize the CNN.

        Args:
            in_channels: Number of input channels.
            activation: Activation function to use.
            out_channels: List of output channels for each convolutional layer.
            kernel_size: List of kernel sizes for each convolutional layer or a single kernel size for all layers.
            stride: List of strides for each convolutional layer or a single stride for all layers.
            flatten: Whether to flatten the output tensor.
            avg_pool: If specified, applies an adaptive average pooling to the given output size after the convolutions.
            batchnorm: Whether to apply batch normalization after each convolutional layer.
            max_pool: Whether to apply max pooling after each convolutional layer.

        .. note::
            Do not save config to allow for the model to be jit compiled.
        """
        super().__init__()

        # If parameters are not lists, convert them to lists
        if isinstance(batchnorm, bool):
            batchnorm = [batchnorm] * len(out_channels)
        if isinstance(max_pool, bool):
            max_pool = [max_pool] * len(out_channels)
        if isinstance(kernel_size, tuple):
            kernel_size = [kernel_size] * len(out_channels)
        if isinstance(stride, int):
            stride = [stride] * len(out_channels)

        # Resolve activation function
        activation_function = resolve_nn_activation(activation)

        # Create layers sequentially
        layers = []
        for idx in range(len(out_channels)):
            in_channels = in_channels if idx == 0 else out_channels[idx - 1]
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels[idx],
                    kernel_size=kernel_size[idx],
                    stride=stride[idx],
                )
            )
            if batchnorm[idx]:
                layers.append(nn.BatchNorm2d(num_features=out_channels[idx]))
            layers.append(activation_function)
            if max_pool[idx]:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

        # Add avgpool if specified
        if avg_pool is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(avg_pool)
        else:
            self.avgpool = None

        # Save flatten flag for forward function
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x)
        if self.flatten:
            x = x.flatten(start_dim=1)
        elif self.avgpool is not None:
            x = self.avgpool(x)
            x = x.flatten(start_dim=1)
        return x

    def init_weights(self) -> None:
        """Initialize the weights of the CNN with Xavier initialization."""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
