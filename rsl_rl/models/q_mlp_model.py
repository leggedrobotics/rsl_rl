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
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        network_cfg: dict | None = None,
    ) -> None:
        """Initialize the Q model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "critic").
            output_dim: Dimension of the output.
            action_dim: Dimension of the actions.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Output distribution configuration. Only ``None`` is supported.
            network_cfg: Configuration dictionary for an alternative network. ``None`` uses the default MLP.
        """
        if distribution_cfg is not None:
            raise ValueError("The QMLPModel does not support an output distribution.")
        self.action_dim = action_dim
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            network_cfg=network_cfg,
        )

    def forward(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass of the state-action value model."""
        latent = torch.cat([self.get_latent(obs), actions], dim=-1)
        return self.mlp(latent)

    def as_jit(self) -> torch.nn.Module:
        """TorchScript export is not implemented for QMLPModel."""
        raise NotImplementedError("QMLPModel does not support TorchScript export.")

    def as_onnx(self) -> torch.nn.Module:
        """ONNX export is not implemented for QMLPModel."""
        raise NotImplementedError("QMLPModel does not support ONNX export.")

    def _get_latent_dim(self) -> int:
        """Return the combined observation and action dimension for the MLP head."""
        return self.obs_dim + self.action_dim
