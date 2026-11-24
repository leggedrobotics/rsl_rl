# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learned physical increments and optional successor-event probabilities."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.algorithms.supermimic_task import State, StateRepresentation
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.utils.utils import remap_state_dict_prefixes


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

        self.input_normalizer = EmpiricalNormalization(self._context_start + (context_dim if normalize_context else 0))
        self.increment_normalizer = EmpiricalNormalization(representation.increment_dim, eps=1e-4)

        self.dynamics_mlp = MLP(
            self._context_start + context_dim,
            representation.increment_dim,
            hidden_dims,
            activation=activation,
            norm="layer",
        )
        if zero_initialize_output:
            nn.init.zeros_(self.dynamics_mlp.output_layer.weight)
            nn.init.zeros_(self.dynamics_mlp.output_layer.bias)

        self.label_head = (
            MLP(feature_dim + context_dim, num_labels, label_hidden_dims, norm="layer") if num_labels else None
        )

    def forward(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> State:
        """Integrate the predicted increment while preserving state and action derivatives."""
        inputs = self._concatenate_dynamics_inputs(state, actions, context)
        if self.normalize_context:
            inputs = self.input_normalizer(inputs)
        else:
            inputs = torch.cat((self.input_normalizer(inputs[..., : self._context_start]), context), dim=-1)

        increment = self.dynamics_mlp(inputs).float()
        if self.normalize_output:
            increment = self.increment_normalizer.inverse(increment)
        with torch.autocast(increment.device.type, enabled=False):
            return self.representation.integrate(state, increment)

    def update_normalization(
        self, state: State, actions: torch.Tensor, context: torch.Tensor, successor: State
    ) -> None:
        """Update real-transition input and increment statistics at collection time."""
        inputs = self._concatenate_dynamics_inputs(state, actions, context)
        self.input_normalizer.update(inputs if self.normalize_context else inputs[..., : self._context_start])
        self.increment_normalizer.update(self.representation.increments(state, successor))

    def predict_label_logits(self, state: State, context: torch.Tensor) -> torch.Tensor:
        """Predict successor events using the matching physical/context input moments."""
        if self.label_head is None:
            raise ValueError("The world model has no event label head.")

        mean = self.input_normalizer.mean
        std = self.input_normalizer.std
        eps = self.input_normalizer.eps
        features = self.representation.features(state)
        features = (features - mean[: self.feature_dim]) / (std[: self.feature_dim] + eps)
        if self.normalize_context:
            context = (context - mean[self._context_start :]) / (std[self._context_start :] + eps)
        return self.label_head(torch.cat((features, context), dim=-1))

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Load current world-model weights or migrate earlier network and normalizer keys."""
        state_dict = remap_state_dict_prefixes(
            state_dict, {"in_norm": "input_normalizer", "out_norm": "increment_normalizer", "net": "dynamics_mlp"}
        )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _concatenate_dynamics_inputs(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Concatenate the physical features, actions and context for dynamics prediction."""
        return torch.cat((self.representation.features(state), actions, context), dim=-1)
