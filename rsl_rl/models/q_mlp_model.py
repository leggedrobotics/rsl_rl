# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel


class QMLPModel(MLPModel):
    """MLP-based state-action value model.

    The model concatenates the (optionally normalized) observations with the actions and maps them to an output of
    dimension ``output_dim``. This is a scalar Q-value for ``output_dim=1`` or, e.g., the logits of a categorical value
    distribution.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        action_dim: int,
        **kwargs: object,
    ) -> None:
        """Initialize the Q model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "critic").
            output_dim: Dimension of the output.
            action_dim: Dimension of the actions.
            **kwargs: Additional arguments passed to :class:`MLPModel`.
        """
        if kwargs.get("distribution_cfg") is not None:
            raise ValueError("The Q model does not support an output distribution.")
        self.action_dim = action_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)  # type: ignore

    def forward(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return the output for the given observations and actions."""
        latent = torch.cat([self.get_latent(obs), actions], dim=-1)
        return self.mlp(latent)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head (observations and actions)."""
        return self.obs_dim + self.action_dim
