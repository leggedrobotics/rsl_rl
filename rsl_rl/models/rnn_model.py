# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState


class RNNModel(MLPModel):
    """RNN-based neural model.

    This model uses a recurrent neural network (RNN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available RNN types are "lstm" and "gru". Observations can be normalized before being passed to
    the RNN. The output of the model can be either deterministic or stochastic, in which case a Gaussian distribution is
    used to sample the outputs.
    """

    is_recurrent: bool = True

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
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
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
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
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

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRNNModel(self, verbose)

    def _get_latent_dim(self) -> int:
        return self.latent_dim


class _TorchGRUModel(nn.Module):
    """Exportable GRU model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        if self.state_dependent_std:
            return out[..., 0, :]
        return out

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMModel(nn.Module):
    """Exportable LSTM model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        if self.state_dependent_std:
            return out[..., 0, :]
        return out

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: RNNModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std

        # Detect RNN type
        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = self.obs_normalizer(obs)

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            out = self.mlp(x)
            if self.state_dependent_std:
                return out[..., 0, :], h, c
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            out = self.mlp(x)

            if self.state_dependent_std:
                return out[..., 0, :], h, None
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
