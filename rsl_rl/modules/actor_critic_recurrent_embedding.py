# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class ActorCriticRecurrentEmbeddings(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_proprio_obs,
        num_critic_proprio_obs,
        num_privileged_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        embd_mlp_dims=[128,64,32],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation_fn = resolve_nn_activation(activation)
        self.num_privileged_obs = num_privileged_obs
        # Define embedding model
        self.priv_mlp = nn.Sequential(
                nn.Linear(num_privileged_obs, embd_mlp_dims[0]),
                activation_fn,
                *[
                layer
                for dim in zip(embd_mlp_dims[:-1], embd_mlp_dims[1:])
                for layer in (nn.Linear(dim[0], dim[1]), activation_fn)
            ]
        )

        num_actor_obs = num_actor_proprio_obs + embd_mlp_dims[-1]
        num_critic_obs = num_critic_proprio_obs + embd_mlp_dims[-1]
        # Define RNN layer
        self.rnn_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.rnn_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Embedding MLP: {self.priv_mlp}")
        print(f"Actor RNN: {self.rnn_a}")
        print(f"Critic RNN: {self.rnn_c}")

    def reset(self, dones=None):
        self.rnn_a.reset(dones)
        self.rnn_c.reset(dones)


    def act(self, observations, masks=None, hidden_states=None):
        # observation: [batch_size, obs_dim] or [seq_len, batch_size, obs_dim]
        proprio_obs = observations[..., :-self.num_privileged_obs]      # [B, 55] or [T, B, 55]
        priv_obs = observations[..., -self.num_privileged_obs : ]       # [B, 4]  or [T, B, 4]

        embd_obs = self.priv_mlp(priv_obs)                              # [B, 32] or [T, B, 32]
        concat_obs = torch.cat([embd_obs, proprio_obs], dim=-1)         # [B, 87] or [T, B, 87]

        input_a = self.rnn_a(concat_obs, masks, hidden_states)          # [1, B, 256] or [1, T*B, 256]
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        proprio_obs = observations[..., :-self.num_privileged_obs]      # [B, 55] or [T, B, 55]
        priv_obs = observations[..., -self.num_privileged_obs : ]       # [B, 4]  or [T, B, 4]

        embd_obs = self.priv_mlp(priv_obs)
        concat_obs = torch.cat([embd_obs, proprio_obs], dim=-1)

        input_a = self.rnn_a(concat_obs)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        proprio_obs = critic_observations[..., :-self.num_privileged_obs]      # [B, 55] or [T, B, 55]
        priv_obs = critic_observations[..., -self.num_privileged_obs : ]       # [B, 4]  or [T, B, 4]

        embd_obs = self.priv_mlp(priv_obs)
        concat_obs = torch.cat([embd_obs, proprio_obs], dim=-1)

        input_c = self.rnn_c(concat_obs, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.rnn_a.hidden_states, self.rnn_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # input shape: [T, B, 87]
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            hidden_state[..., dones == 1, :] = 0.0
