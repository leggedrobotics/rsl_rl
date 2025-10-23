# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization

from .actor_critic import ActorCritic


class ActorCriticPerceptive(ActorCritic):
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        actor_cnn_config: dict[str, dict] | dict | None = None,
        critic_cnn_config: dict[str, dict] | dict | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "PerceptiveActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        nn.Module.__init__(self)

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        num_actor_in_channels = []
        self.actor_obs_group_1d = []
        self.actor_obs_group_2d = []
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.actor_obs_group_2d.append(obs_group)
                num_actor_in_channels.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.actor_obs_group_1d.append(obs_group)
                num_actor_obs += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        self.critic_obs_group_1d = []
        self.critic_obs_group_2d = []
        num_critic_obs = 0
        num_critic_in_channels = []
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                self.critic_obs_group_2d.append(obs_group)
                num_critic_in_channels.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                self.critic_obs_group_1d.append(obs_group)
                num_critic_obs += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        # actor cnn
        if self.actor_obs_group_2d:
            assert actor_cnn_config is not None, "Actor CNN config is required for 2D actor observations."

            # check if multiple 2D actor observations are provided
            if len(self.actor_obs_group_2d) > 1 and all(isinstance(item, dict) for item in actor_cnn_config.values()):
                assert len(actor_cnn_config) == len(self.actor_obs_group_2d), (
                    "Number of CNN configs must match number of 2D actor observations."
                )
            elif len(self.actor_obs_group_2d) > 1:
                print(
                    "Only one CNN config for multiple 2D actor observations given, using the same CNN for all groups."
                )
                actor_cnn_config = dict(zip(self.actor_obs_group_2d, [actor_cnn_config] * len(self.actor_obs_group_2d)))
            else:
                actor_cnn_config = dict(zip(self.actor_obs_group_2d, [actor_cnn_config]))

            self.actor_cnns = nn.ModuleDict()
            encoding_dims = []
            for idx, obs_group in enumerate(self.actor_obs_group_2d):
                self.actor_cnns[obs_group] = CNN(num_actor_in_channels[idx], activation, **actor_cnn_config[obs_group])
                print(f"Actor CNN for {obs_group}: {self.actor_cnns[obs_group]}")

                # compute the encoding dimension (cpu necessary as model not moved to device yet)
                encoding_dims.append(self.actor_cnns[obs_group](obs[obs_group].to("cpu")).shape[-1])

            encoding_dim = sum(encoding_dims)
        else:
            self.actor_cnns = None
            encoding_dim = 0

        # actor mlp
        self.actor = MLP(num_actor_obs + encoding_dim, num_actions, actor_hidden_dims, activation)

        # actor observation normalization (only for 1D actor observations)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic cnn
        if self.critic_obs_group_2d:
            assert critic_cnn_config is not None, "Critic CNN config is required for 2D critic observations."

            # check if multiple 2D critic observations are provided
            if len(self.critic_obs_group_2d) > 1 and all(isinstance(item, dict) for item in critic_cnn_config.values()):
                assert len(critic_cnn_config) == len(self.critic_obs_group_2d), (
                    "Number of CNN configs must match number of 2D critic observations."
                )
            elif len(self.critic_obs_group_2d) > 1:
                print(
                    "Only one CNN config for multiple 2D critic observations given, using the same CNN for all groups."
                )
                critic_cnn_config = dict(
                    zip(self.critic_obs_group_2d, [critic_cnn_config] * len(self.critic_obs_group_2d))
                )
            else:
                critic_cnn_config = dict(zip(self.critic_obs_group_2d, [critic_cnn_config]))

            self.critic_cnns = nn.ModuleDict()
            encoding_dims = []
            for idx, obs_group in enumerate(self.critic_obs_group_2d):
                self.critic_cnns[obs_group] = CNN(
                    num_critic_in_channels[idx], activation, **critic_cnn_config[obs_group]
                )
                print(f"Critic CNN for {obs_group}: {self.critic_cnns[obs_group]}")

                # compute the encoding dimension (cpu necessary as model not moved to device yet)
                encoding_dims.append(self.critic_cnns[obs_group](obs[obs_group].to("cpu")).shape[-1])

            encoding_dim = sum(encoding_dims)
        else:
            self.critic_cnns = None
            encoding_dim = 0

        # critic mlp
        self.critic = MLP(num_critic_obs + encoding_dim, 1, critic_hidden_dims, activation)

        # critic observation normalization (only for 1D critic observations)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution: Normal = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        if self.actor_cnns is not None:
            # encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_group_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # update mlp obs
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        super().update_distribution(mlp_obs)

    def act(self, obs, **kwargs):
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)
        self.update_distribution(mlp_obs, cnn_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        mlp_obs, cnn_obs = self.get_actor_obs(obs)
        mlp_obs = self.actor_obs_normalizer(mlp_obs)

        if self.actor_cnns is not None:
            # encode the 2D actor observations
            cnn_enc_list = [self.actor_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.actor_obs_group_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # update mlp obs
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        return self.actor(mlp_obs)

    def evaluate(self, obs, **kwargs):
        mlp_obs, cnn_obs = self.get_critic_obs(obs)
        mlp_obs = self.critic_obs_normalizer(mlp_obs)

        if self.critic_cnns is not None:
            # encode the 2D critic observations
            cnn_enc_list = [self.critic_cnns[obs_group](cnn_obs[obs_group]) for obs_group in self.critic_obs_group_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            # update mlp obs
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)

        return self.critic(mlp_obs)

    def get_actor_obs(self, obs):
        obs_dict_2d = {}
        obs_list_1d = [obs[obs_group] for obs_group in self.actor_obs_group_1d]
        for obs_group in self.actor_obs_group_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def get_critic_obs(self, obs):
        obs_dict_2d = {}
        obs_list_1d = [obs[obs_group] for obs_group in self.critic_obs_group_1d]
        for obs_group in self.critic_obs_group_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        return torch.cat(obs_list_1d, dim=-1), obs_dict_2d

    def update_normalization(self, obs) -> None:
        if self.actor_obs_normalization:
            actor_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)
