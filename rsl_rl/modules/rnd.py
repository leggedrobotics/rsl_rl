# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.normalizer import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class RandomNetworkDistillation(nn.Module):
    """Implementation of Random Network Distillation (RND) [1]

    References:
        .. [1] Burda, Yuri, et al. "Exploration by random network distillation." arXiv preprint arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        num_states: int,
        num_outputs: int,
        predictor_hidden_dims: list[int],
        target_hidden_dims: list[int],
        activation: str = "elu",
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = "cpu",
        weight_schedule: dict | None = None,
    ):
        """Initialize the RND module.

        - If :attr:`state_normalization` is True, then the input state is normalized using an Empirical Normalization layer.
        - If :attr:`reward_normalization` is True, then the intrinsic reward is normalized using an Empirical Discounted
          Variation Normalization layer.

        .. note::
            If the hidden dimensions are -1 in the predictor and target networks configuration, then the number of states
            is used as the hidden dimension.

        Args:
            num_states: Number of states/inputs to the predictor and target networks.
            num_outputs: Number of outputs (embedding size) of the predictor and target networks.
            predictor_hidden_dims: List of hidden dimensions of the predictor network.
            target_hidden_dims: List of hidden dimensions of the target network.
            activation: Activation function. Defaults to "elu".
            weight: Scaling factor of the intrinsic reward. Defaults to 0.0.
            state_normalization: Whether to normalize the input state. Defaults to False.
            reward_normalization: Whether to normalize the intrinsic reward. Defaults to False.
            device: Device to use. Defaults to "cpu".
            weight_schedule: The type of schedule to use for the RND weight parameter.
                Defaults to None, in which case the weight parameter is constant.
                It is a dictionary with the following keys:

                - "mode": The type of schedule to use for the RND weight parameter.
                    - "constant": Constant weight schedule.
                    - "step": Step weight schedule.
                    - "linear": Linear weight schedule.

                For the "step" weight schedule, the following parameters are required:

                - "final_step": The step at which the weight parameter is set to the final value.
                - "final_value": The final value of the weight parameter.

                For the "linear" weight schedule, the following parameters are required:
                - "initial_step": The step at which the weight parameter is set to the initial value.
                - "final_step": The step at which the weight parameter is set to the final value.
                - "final_value": The final value of the weight parameter.
        """
        # initialize parent class
        super().__init__()

        # Store parameters
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.initial_weight = weight
        self.device = device
        self.state_normalization = state_normalization
        self.reward_normalization = reward_normalization

        # Normalization of input gates
        if state_normalization:
            self.state_normalizer = EmpiricalNormalization(shape=[self.num_states], until=1.0e8).to(self.device)
        else:
            self.state_normalizer = torch.nn.Identity()
        # Normalization of intrinsic reward
        if reward_normalization:
            self.reward_normalizer = EmpiricalDiscountedVariationNormalization(shape=[], until=1.0e8).to(self.device)
        else:
            self.reward_normalizer = torch.nn.Identity()

        # counter for the number of updates
        self.update_counter = 0

        # resolve weight schedule
        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            self.weight_scheduler = getattr(self, f"_{weight_schedule['mode']}_weight_schedule")
        else:
            self.weight_scheduler = None
        # Create network architecture
        self.predictor = self._build_mlp(num_states, predictor_hidden_dims, num_outputs, activation).to(self.device)
        self.target = self._build_mlp(num_states, target_hidden_dims, num_outputs, activation).to(self.device)

        # make target network not trainable
        self.target.eval()

    def get_intrinsic_reward(self, rnd_state) -> tuple[torch.Tensor, torch.Tensor]:
        # note: the counter is updated number of env steps per learning iteration
        self.update_counter += 1
        # Normalize rnd state
        rnd_state = self.state_normalizer(rnd_state)
        # Obtain the embedding of the rnd state from the target and predictor networks
        target_embedding = self.target(rnd_state).detach()
        predictor_embedding = self.predictor(rnd_state).detach()
        # Compute the intrinsic reward as the distance between the embeddings
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # Normalize intrinsic reward
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        # Check the weight schedule
        if self.weight_scheduler is not None:
            self.weight = self.weight_scheduler(step=self.update_counter, **self.weight_scheduler_params)
        else:
            self.weight = self.initial_weight
        # Scale intrinsic reward
        intrinsic_reward *= self.weight

        return intrinsic_reward, rnd_state

    def forward(self, *args, **kwargs):
        raise RuntimeError("Forward method is not implemented. Use get_intrinsic_reward instead.")

    def train(self, mode: bool = True):
        # sets module into training mode
        self.predictor.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        if self.reward_normalization:
            self.reward_normalizer.train(mode)
        return self

    def eval(self):
        return self.train(False)

    """
    Private Methods
    """

    @staticmethod
    def _build_mlp(input_dims: int, hidden_dims: list[int], output_dims: int, activation_name: str = "elu"):
        """Builds target and predictor networks"""

        network_layers = []
        # resolve hidden dimensions
        # if dims is -1 then we use the number of observations
        hidden_dims = [input_dims if dim == -1 else dim for dim in hidden_dims]
        # resolve activation function
        activation = resolve_nn_activation(activation_name)
        # first layer
        network_layers.append(nn.Linear(input_dims, hidden_dims[0]))
        network_layers.append(activation)
        # subsequent layers
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                # last layer
                network_layers.append(nn.Linear(hidden_dims[layer_index], output_dims))
            else:
                # hidden layers
                network_layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                network_layers.append(activation)
        return nn.Sequential(*network_layers)

    """
    Different weight schedules.
    """

    def _constant_weight_schedule(self, step: int, **kwargs):
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs):
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs):
        if step < initial_step:
            return self.initial_weight
        elif step > final_step:
            return final_value
        else:
            return self.initial_weight + (final_value - self.initial_weight) * (step - initial_step) / (
                final_step - initial_step
            )
