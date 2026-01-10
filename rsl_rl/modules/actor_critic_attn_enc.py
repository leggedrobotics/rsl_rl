# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, AttentionEncoder


class ActorCriticAttnEnc(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        embedding_dim: int = 64,
        head_num: int = 8,
        map_size: tuple[int] = (17, 11),
        map_resolution: float = 0.1,
        single_obs_dim: int = 78,
        critic_estimation: bool = False,
        estimation_slice:list [int] = [78, 79, 80],
        estimator_hidden_dims: tuple[int] | list[int] = [256, 64],
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticAttnEnc.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        self.num_actor_obs = 0
        self.map_size = map_size
        self.single_obs_dim = single_obs_dim
        self.critic_estimation = critic_estimation
        self.num_estimation = len(estimation_slice)
        self.estimation_slice = estimation_slice
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            self.num_actor_obs += obs[obs_group].shape[-1]
        self.num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            self.num_critic_obs += obs[obs_group].shape[-1]

        if self.critic_estimation:
            self.last_critic_pred: torch.Tensor = None
            self.last_critic_gt: torch.Tensor = None

        # Encoder
        if self.critic_estimation:
            self.encoder =  AttentionEncoder(single_obs_dim+self.num_estimation, embedding_dim, head_num, self.map_size, map_resolution)
        else:
            self.encoder =  AttentionEncoder(single_obs_dim, embedding_dim, head_num, self.map_size, map_resolution)
        print(f"Encoder : {self.encoder}")

        if self.critic_estimation:
            self.estimator = MLP(self.num_actor_obs, self.num_estimation, estimator_hidden_dims, activation)
            mlp_input_dim_a = embedding_dim + single_obs_dim + self.num_estimation
            print(f"Estimator : {self.estimator}")
        else:
            mlp_input_dim_a = embedding_dim + self.num_actor_obs
        
        mlp_input_dim_c = embedding_dim + self.num_critic_obs

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(mlp_input_dim_a, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(mlp_input_dim_a, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(mlp_input_dim_c, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs, perception_obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.critic_estimation:
            self.last_critic_pred = self.estimator(obs)
            obs = torch.cat([obs[:, -self.single_obs_dim:], self.last_critic_pred], dim=-1)
            embedding, *_ = self.encoder(obs, perception_obs)
        else:
            embedding, *_ = self.encoder(obs[:, -self.single_obs_dim:], perception_obs)
        obs = torch.cat([obs, embedding], dim=-1)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, return_attention: bool = False) -> torch.Tensor:
        obs, perception_obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.critic_estimation:
            critic_pred = self.estimator(obs)
            obs = torch.cat([obs[:, -self.single_obs_dim:], critic_pred], dim=-1)
            embedding, attention = self.encoder(obs, perception_obs)
        else:
            embedding, attention = self.encoder(obs[:, -self.single_obs_dim:], perception_obs)
        obs = torch.cat([obs, embedding], dim=-1)
        if self.state_dependent_std:
            actions = self.actor(obs)[..., 0, :]
        else:
            actions = self.actor(obs)
        if return_attention:
            return actions, attention
        else:
            return actions

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs, perception_obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        if self.critic_estimation:
            self.last_critic_gt = obs[:, self.estimation_slice]
            embedding, *_ = self.encoder(torch.cat([obs[:, :self.single_obs_dim], self.last_critic_gt], dim=-1), perception_obs)
        else:
            embedding, *_ = self.encoder(obs[:, :self.single_obs_dim], perception_obs)
        obs = torch.cat([obs, embedding], dim=-1)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        perception_obs = obs["perception_a"] if "perception_a" in self.obs_groups["perception"] else None
        return torch.cat(obs_list, dim=-1), perception_obs

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        perception_obs = obs["perception_c"] if "perception_c" in self.obs_groups["perception"] else None
        return torch.cat(obs_list, dim=-1), perception_obs

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs, *_ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, *_ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_estimation_loss(self) -> torch.Tensor | None:
        if self.critic_estimation and self.last_critic_pred is not None and self.last_critic_gt is not None:
            critic_mse_loss = torch.nn.MSELoss()
            critic_loss = critic_mse_loss(self.last_critic_pred, self.last_critic_gt.detach())
            return critic_loss
        else:
            return None
