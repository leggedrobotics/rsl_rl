# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories

import torch

class ActorCriticRL2(ActorCritic):
    """
    Minimal RL^2 actor-critic:
    - Actor input: obs + prev_action
    - Critic input: obs
    """
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated, use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRL2.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys())
            )

        # ActorCritic base expects num_actor_obs = rnn_hidden_dim
        super().__init__(
            num_actor_obs=rnn_hidden_dim + num_actor_obs,   # 现在的实验条件默认critic_obs=actor_obs
            num_critic_obs=rnn_hidden_dim + num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        # Actor RNN input = obs + prev_action
        self.memory_a = Memory(
            input_size=num_actor_obs+num_actions,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )
        # Critic RNN input = critic_obs + prev_action
        self.memory_c = Memory(
            input_size=num_critic_obs+num_actions,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    # def act(self, observations, prev_actions, masks=None, hidden_states=None):
    #     # concat obs + prev_action along last dim
    #     if hidden_states is not None and observations.dim() == 2:
    #         # add dumping time dimension
    #         observations = observations.unsqueeze(0)
    #         prev_actions = prev_actions.unsqueeze(0)
    #     # input_a = torch.cat([observations, prev_actions], dim=-1)
    #     input_a = observations
    #     input_a = self.memory_a(input_a, masks, hidden_states, chunk_mode=True)
    #     input_a = input_a.squeeze(0)
    #     # 如果 observations 和 input_a 维度不一致, 且用mask掩码处理
    #     if input_a.shape[:-1] != observations.shape[:-1] and masks is not None:
    #         # self.alg.update()的时候可能出现以上情况
    #         # masked_obs = unpad_trajectories(observations, masks)
    #         # masked_obs = masked_obs.squeeze(0)
    #         # input_a = torch.cat([input_a, masked_obs], dim=-1)
    #         masked_obs = observations.squeeze(0)
    #     else:
    #         # input_a = torch.cat([input_a, observations], dim=-1)
    #         masked_obs = observations  # (num_envs, obs_dim)
    #     return super().act(masked_obs)

    def act(self, observations, prev_actions, masks=None, hidden_states=None):
        if masks is not None:
            pass
        input_a = torch.cat([observations, prev_actions], dim=-1)
        input_a = self.memory_a(input_a, masks, hidden_states)
        mlp_a_input = torch.cat([input_a.squeeze(0), observations], dim=-1)
        return super().act(mlp_a_input)

    # 脚本训练过程用不到，应该不影响训练，暂时不修改
    def act_inference(self, observations, prev_actions):
        input_a = torch.cat([observations, prev_actions], dim=-1)
        input_a = self.memory_a(input_a)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, prev_action, masks=None, hidden_states=None):
        input_c = torch.cat([critic_observations, prev_action], dim=-1)
        # actor和critic共用一个RNN
        input_c = self.memory_a(input_c, masks, hidden_states)
        mlp_c_input = torch.cat([input_c.squeeze(0), critic_observations], dim=-1)
        return super().evaluate(mlp_c_input)

    # # 我们改成critic和actor使用同一个RNN，输入相同context和obs拼接
    # def evaluate(self, observations, prev_actions, masks=None, hidden_states=None):
    #     # concat obs + prev_action along last dim
    #     if hidden_states is not None and observations.dim() == 2:
    #         # add dumping time dimension
    #         observations = observations.unsqueeze(0)
    #         prev_actions = prev_actions.unsqueeze(0)
    #     # input_a = torch.cat([observations, prev_actions], dim=-1)
    #     input_a = observations
    #     input_a = self.memory_c(input_a, masks, hidden_states, chunk_mode=True)
    #     input_a = input_a.squeeze(0)
    #     # 如果 observations 和 input_a 维度不一致, 用mask掩码处理
    #     if input_a.shape[:-1] != observations.shape[:-1] and masks is not None:
    #         # masked_obs = unpad_trajectories(observations, masks)
    #         # masked_obs = masked_obs.squeeze(0)
    #         # input_a = torch.cat([input_a, masked_obs], dim=-1)
    #         # masked_obs = observations
    #         masked_obs = observations.squeeze(0)
    #     else:
    #         # input_a = torch.cat([input_a, observations], dim=-1)
    #         masked_obs = observations
    #     return super().evaluate(masked_obs)

    # 强制都返回Mem_a的隐层，和上层的API对齐，减少修改
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_a.hidden_states