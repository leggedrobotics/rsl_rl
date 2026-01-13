# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warnings
from tensordict import TensorDict
from typing import Any

from rsl_rl.models import MLPModel
from rsl_rl.modules import RNN, HiddenState


class RNNModel(MLPModel):
    """RNN-based neural model.

    This model uses a recurrent neural network (RNN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available RNN types are "lstm" and "gru". Observations can be normalized before being passed to
    the RNN. The output of the model can be either deterministic or stochastic, in which case a Gaussian distribution is
    used to sample the outputs.
    """

    is_recurrent: bool = True

    def __init__(  # noqa: D417
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        obs_normalization: bool = False,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            stochastic: Whether the model outputs stochastic or deterministic values.
            init_noise_std: Initial standard deviation of the stochatic output.
            noise_std_type: Whether the standard deviation is defined as a "scalar" or in "log" space.
            state_dependent_std: Whether the standard deviation is state dependent.
            rnn_type: Type of RNN to use ("lstm" or "gru").
            rnn_hidden_dim: Dimension of the RNN hidden state.
            rnn_num_layers: Number of RNN layers.
        """
        # Handle deprecated argument
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")  # type: ignore
        self.latent_dim = rnn_hidden_dim

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            stochastic,
            init_noise_std,
            noise_std_type,
            state_dependent_std,
        )

        # RNN
        self.rnn = RNN(self.obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        # Extract and concatenate observation groups and normalize
        latent = super().get_latent(obs)
        # Pass through the RNN
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state  # type: ignore

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)

    def _get_latent_dim(self) -> int:
        return self.latent_dim
