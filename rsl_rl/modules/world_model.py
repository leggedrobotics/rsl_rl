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
    """Predict the next physical state and optional event labels."""

    def __init__(
        self,
        representation: StateRepresentation,
        feature_dim: int,
        num_actions: int,
        context_dim: int,
        hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        num_labels: int = 0,
        label_hidden_dims: tuple[int, ...] = (512, 512),
        activation: str = "swish",
        normalize_output: bool = True,
        normalize_context: bool = True,
        zero_initialize_output: bool = True,
    ) -> None:
        """Build normalized residual dynamics and an optional event head."""
        super().__init__()
        self.representation = representation
        self.feature_dim = feature_dim
        self.normalize_output = normalize_output
        self.normalize_context = normalize_context
        self._context_start = feature_dim + num_actions

        self.in_norm = EmpiricalNormalization(self._context_start + (context_dim if normalize_context else 0))
        self.out_norm = EmpiricalNormalization(representation.increment_dim, eps=1e-4)
        self.net = MLP(
            self._context_start + context_dim,
            representation.increment_dim,
            hidden_dims,
            activation=activation,
            norm="layer",
        )
        if zero_initialize_output:
            nn.init.zeros_(self.net.output_layer.weight)
            nn.init.zeros_(self.net.output_layer.bias)
        self.label_head = (
            MLP(feature_dim + context_dim, num_labels, label_hidden_dims, norm="layer") if num_labels else None
        )

    def forward(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> State:
        """Integrate the predicted increment while preserving state and action derivatives."""
        inputs = self._dynamics_inputs(state, actions, context)
        if self.normalize_context:
            inputs = self.in_norm(inputs)
        else:
            inputs = torch.cat((self.in_norm(inputs[..., : self._context_start]), context), dim=-1)
        increment = self.net(inputs).float()
        if self.normalize_output:
            increment = self.out_norm.inverse(increment)
        with torch.autocast(increment.device.type, enabled=False):
            return self.representation.integrate(state, increment)

    def observe(self, state: State, actions: torch.Tensor, context: torch.Tensor, successor: State) -> None:
        """Update real-transition input and increment statistics at collection time."""
        inputs = self._dynamics_inputs(state, actions, context)
        self.in_norm.update(inputs if self.normalize_context else inputs[..., : self._context_start])
        self.out_norm.update(self.representation.increments(state, successor))

    def label_logits(self, state: State, context: torch.Tensor) -> torch.Tensor:
        """Predict successor events using the matching physical/context input moments."""
        if self.label_head is None:
            raise ValueError("The world model has no event label head.")
        mean = self.in_norm.mean
        std = self.in_norm.std
        features = self.representation.features(state)
        features = (features - mean[: self.feature_dim]) / (std[: self.feature_dim] + self.in_norm.eps)
        if self.normalize_context:
            context = (context - mean[self._context_start :]) / (std[self._context_start :] + self.in_norm.eps)
        return self.label_head(torch.cat((features, context), dim=-1))

    def _dynamics_inputs(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Concatenate the physical features, actions and context for dynamics prediction."""
        return torch.cat((self.representation.features(state), actions, context), dim=-1)
