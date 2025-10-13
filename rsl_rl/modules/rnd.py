# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, NoReturn

from rsl_rl.env import VecEnv
from rsl_rl.networks import MLP, EmpiricalDiscountedVariationNormalization, EmpiricalNormalization


class RandomNetworkDistillation(nn.Module):
    """Implementation of Random Network Distillation (RND) [1].

    References:
        .. [1] Burda, Yuri, et al. "Exploration by Random Network Distillation." arXiv preprint arXiv:1810.12894 (2018).
    """

    def __init__(
        self,
        num_states: int,
        obs_groups: dict,
        num_outputs: int,
        predictor_hidden_dims: tuple[int] | list[int],
        target_hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = "cpu",
        weight_schedule: dict | None = None,
    ) -> None:
        """Initialize the RND module.

        - If :attr:`state_normalization` is True, then the input state is normalized using an Empirical Normalization
          layer.
        - If :attr:`reward_normalization` is True, then the intrinsic reward is normalized using an Empirical Discounted
          Variation Normalization layer.
        - If the hidden dimensions are -1 in the predictor and target networks configuration, then the number of states
          is used as the hidden dimension.

        Args:
            num_states: Number of states/inputs to the predictor and target networks.
            obs_groups: Dictionary of observation groups.
            num_outputs: Number of outputs (embedding size) of the predictor and target networks.
            predictor_hidden_dims: List of hidden dimensions of the predictor network.
            target_hidden_dims: List of hidden dimensions of the target network.
            activation: Activation function.
            weight: Scaling factor of the intrinsic reward.
            state_normalization: Whether to normalize the input state.
            reward_normalization: Whether to normalize the intrinsic reward.
            device: Device to use.
            weight_schedule: Type of schedule to use for the RND weight parameter.
                It is a dictionary with the following keys:

                - "mode": Type of schedule to use for the RND weight parameter.
                    - "constant": Constant weight schedule.
                    - "step": Step weight schedule.
                    - "linear": Linear weight schedule.

                For the "step" weight schedule, the following parameters are required:

                - "final_step": Step at which the weight parameter is set to the final value.
                - "final_value": Final value of the weight parameter.

                For the "linear" weight schedule, the following parameters are required:
                - "initial_step": Step at which the weight parameter is set to the initial value.
                - "final_step": Step at which the weight parameter is set to the final value.
                - "final_value": Final value of the weight parameter.
        """
        # Initialize parent class
        super().__init__()

        # Store parameters
        self.num_states = num_states
        self.obs_groups = obs_groups
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

        # Counter for the number of updates
        self.update_counter = 0

        # Resolve weight schedule
        if weight_schedule is not None:
            self.weight_scheduler_params = weight_schedule
            self.weight_scheduler = getattr(self, f"_{weight_schedule['mode']}_weight_schedule")
        else:
            self.weight_scheduler = None

        # Create network architecture
        self.predictor = MLP(num_states, num_outputs, predictor_hidden_dims, activation).to(self.device)
        self.target = MLP(num_states, num_outputs, target_hidden_dims, activation).to(self.device)

        # Make target network not trainable
        self.target.eval()

    def get_intrinsic_reward(self, obs: TensorDict) -> torch.Tensor:
        # Note: The counter is updated number of env steps per learning iteration
        self.update_counter += 1
        # Extract the rnd state from the observation
        rnd_state = self.get_rnd_state(obs)
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

        return intrinsic_reward

    def forward(self, *args: Any, **kwargs: dict[str, Any]) -> NoReturn:
        raise RuntimeError("Forward method is not implemented. Use get_intrinsic_reward instead.")

    def train(self, mode: bool = True) -> RandomNetworkDistillation:
        # Set module into training mode
        self.predictor.train(mode)
        if self.state_normalization:
            self.state_normalizer.train(mode)
        if self.reward_normalization:
            self.reward_normalizer.train(mode)
        return self

    def eval(self) -> RandomNetworkDistillation:
        return self.train(False)

    def get_rnd_state(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["rnd_state"]]
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        # Normalize the state
        if self.state_normalization:
            rnd_state = self.get_rnd_state(obs)
            self.state_normalizer.update(rnd_state)

    def _constant_weight_schedule(self, step: int, **kwargs: dict[str, Any]) -> float:
        return self.initial_weight

    def _step_weight_schedule(self, step: int, final_step: int, final_value: float, **kwargs: dict[str, Any]) -> float:
        return self.initial_weight if step < final_step else final_value

    def _linear_weight_schedule(
        self, step: int, initial_step: int, final_step: int, final_value: float, **kwargs: dict[str, Any]
    ) -> float:
        if step < initial_step:
            return self.initial_weight
        elif step > final_step:
            return final_value
        else:
            return self.initial_weight + (final_value - self.initial_weight) * (step - initial_step) / (
                final_step - initial_step
            )


def resolve_rnd_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]], env: VecEnv) -> dict:
    """Resolve the RND configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        obs: Observation dictionary.
        obs_groups: Observation groups dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # Resolve dimension of rnd gated state
    if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
        # Get dimension of rnd gated state
        num_rnd_state = 0
        for obs_group in obs_groups["rnd_state"]:
            assert len(obs[obs_group].shape) == 2, "The RND module only supports 1D observations."
            num_rnd_state += obs[obs_group].shape[-1]
        # Add rnd gated state to config
        alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
        alg_cfg["rnd_cfg"]["obs_groups"] = obs_groups
        # Scale down the rnd weight with timestep
        alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt
    return alg_cfg
