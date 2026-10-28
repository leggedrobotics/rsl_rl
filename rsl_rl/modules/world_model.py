# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learned physical increments and optional successor-event probabilities."""

from __future__ import annotations

import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.algorithms.supermimic_task import State, StateRepresentation
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class WorldModel(nn.Module):
    """Predict increments in task-owned coordinates and preserve state/action Jacobians."""

    def __init__(
        self,
        representation: StateRepresentation,
        feature_dim: int,
        num_actions: int,
        context_dim: int,
        hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        num_labels: int = 0,
        label_hidden_dims: tuple[int, ...] = (512, 512),
    ) -> None:
        """Build normalized residual dynamics and an optional event head."""
        super().__init__()
        self.representation = representation
        self.feature_dim, self.num_actions = feature_dim, num_actions
        self.in_norm = EmpiricalNormalization(feature_dim + num_actions + context_dim)
        self.out_norm = EmpiricalNormalization(representation.increment_dim, eps=1e-4)
        self.net = MLP(
            feature_dim + num_actions + context_dim,
            representation.increment_dim,
            hidden_dims,
            activation="swish",
            norm="layer",
        )
        nn.init.zeros_(self.net.output.weight)
        nn.init.zeros_(self.net.output.bias)
        self.label_head = (
            MLP(feature_dim + context_dim, num_labels, label_hidden_dims, norm="layer") if num_labels else None
        )

    def inputs(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Concatenate physical state features, actions and task-normalized privileged context."""
        return torch.cat((self.representation.features(state), actions, context), -1)

    def observe(self, state: State, actions: torch.Tensor, context: torch.Tensor, successor: State) -> None:
        """Update real-transition input and increment statistics at collection time."""
        self.in_norm.update(self.inputs(state, actions, context))
        self.out_norm.update(self.representation.increments(state, successor))

    def label_logits(self, state: State, context: torch.Tensor) -> torch.Tensor:
        """Predict successor events using the matching physical/context input moments."""
        if self.label_head is None:
            raise ValueError("The world model has no event label head.")
        start = self.feature_dim + self.num_actions
        mean = torch.cat((self.in_norm.mean[: self.feature_dim], self.in_norm.mean[start:]))
        std = torch.cat((self.in_norm.std[: self.feature_dim], self.in_norm.std[start:])) + self.in_norm.eps
        features = torch.cat((self.representation.features(state), context), -1)
        return self.label_head((features - mean) / std)

    def forward(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> State:
        """Integrate the predicted physical increment without detaching its inputs."""
        increment = self.net(self.in_norm(self.inputs(state, actions, context))).float()
        increment = self.out_norm.inverse(increment)
        return self.representation.integrate(state, increment)
