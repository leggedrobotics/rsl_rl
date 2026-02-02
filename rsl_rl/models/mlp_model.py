# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories


class MLPModel(nn.Module):
    """MLP-based neural model.

    This model uses a simple multi-layer perceptron (MLP) to process 1D observation groups. Obsevations can be
    normalized before being passed to the MLP. The output of the model can be either deterministic or
    stochastic, in which case a Gaussian distribution is used to sample the outputs.
    """

    is_recurrent: bool = False

    def __init__(
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
    ) -> None:
        """Initialize the MLP-based model.

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
        """
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        # MLP
        self.state_dependent_std = state_dependent_std
        if state_dependent_std and stochastic:
            self.mlp = MLP(self._get_latent_dim(), [2, output_dim], hidden_dims, activation)
        else:
            self.mlp = MLP(self._get_latent_dim(), output_dim, hidden_dims, activation)

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if state_dependent_std and stochastic:
            # Initialize weights and biases for the last layer of the std head
            torch.nn.init.zeros_(self.mlp[-2].weight[output_dim:])  # type: ignore
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.mlp[-2].bias[output_dim:], init_noise_std)  # type: ignore
            elif self.noise_std_type == "log":
                init_noise_std_log = torch.log(torch.tensor(init_noise_std + 1e-7))
                torch.nn.init.constant_(self.mlp[-2].bias[output_dim:], init_noise_std_log)  # type: ignore
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        elif stochastic:
            # Initialize parameters for state independent std
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Note: Populated in update_distribution
        self.distribution = None
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model is initialized as stochastic and defaults to
            `False`, meaning that even stochastic models will return deterministic outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        if self.stochastic and stochastic_output:
            self._update_distribution(latent)
            return self.distribution.sample()
        else:
            if self.state_dependent_std:
                return self.mlp(latent)[..., 0, :]
            else:
                return self.mlp(latent)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        latent = self.obs_normalizer(latent)
        return latent

    def get_hidden_state(self) -> HiddenState:
        return None

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs).sum(dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            # Select and concatenate observations
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            # Update the normalizer parameters
            self.obs_normalizer.update(mlp_obs)  # type: ignore

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.mlp(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.mlp(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            assert len(obs[obs_group].shape) == 2, "The MLP model only supports 1D observations."
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim


class _TorchMLPModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: MLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        if self.state_dependent_std:
            return out[..., 0, :]
        return out

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        if self.state_dependent_std:
            return out[..., 0, :]
        return out

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
