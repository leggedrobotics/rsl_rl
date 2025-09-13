# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING, dataclass
from torch import nn as nn

from rsl_rl.utils import resolve_nn_activation


@dataclass
class CNNConfig:
    out_channels: list[int] = MISSING
    kernel_size: list[tuple[int, int]] | tuple[int, int] = MISSING
    stride: list[int] | int = 1
    flatten: bool = True
    avg_pool: tuple[int, int] | None = None
    batchnorm: bool | list[bool] = False
    max_pool: bool | list[bool] = False


class CNN(nn.Module):
    def __init__(self, cfg: CNNConfig, in_channels: int, activation: str):
        """
        Convolutional Neural Network model.

        .. note::
            Do not save config to allow for the model to be jit compiled.
        """
        super().__init__()

        if isinstance(cfg.batchnorm, bool):
            cfg.batchnorm = [cfg.batchnorm] * len(cfg.out_channels)
        if isinstance(cfg.max_pool, bool):
            cfg.max_pool = [cfg.max_pool] * len(cfg.out_channels)
        if isinstance(cfg.kernel_size, tuple):
            cfg.kernel_size = [cfg.kernel_size] * len(cfg.out_channels)
        if isinstance(cfg.stride, int):
            cfg.stride = [cfg.stride] * len(cfg.out_channels)

        # get activation function
        activation_function = resolve_nn_activation(activation)

        # build model layers
        modules = []

        for idx in range(len(cfg.out_channels)):
            in_channels = cfg.in_channels if idx == 0 else cfg.out_channels[idx - 1]
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=cfg.out_channels[idx],
                    kernel_size=cfg.kernel_size[idx],
                    stride=cfg.stride[idx],
                )
            )
            if cfg.batchnorm[idx]:
                modules.append(nn.BatchNorm2d(num_features=cfg.out_channels[idx]))
            modules.append(activation_function)
            if cfg.max_pool[idx]:
                modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.architecture = nn.Sequential(*modules)

        if cfg.avg_pool is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(cfg.avg_pool)
        else:
            self.avgpool = None

        # initialize weights
        self.init_weights(self.architecture)

        # save flatten config for forward function
        self.flatten = cfg.flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.architecture(x)
        if self.flatten:
            x = x.flatten(start_dim=1)
        elif self.avgpool is not None:
            x = self.avgpool(x)
            x = x.flatten(start_dim=1)
        return x

    @staticmethod
    def init_weights(sequential):
        [
            torch.nn.init.xavier_uniform_(module.weight)
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Conv2d))
        ]
