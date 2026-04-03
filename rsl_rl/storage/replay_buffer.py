# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch


class ReplayBuffer:
    """Simple replay buffer for vectorized environment transitions."""

    def __init__(
        self,
        capacity: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.actor_obs = torch.zeros(capacity, actor_obs_dim, device=device)
        self.critic_obs = torch.zeros(capacity, critic_obs_dim, device=device)
        self.next_actor_obs = torch.zeros(capacity, actor_obs_dim, device=device)
        self.next_critic_obs = torch.zeros(capacity, critic_obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)

    def add(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
    ) -> None:
        batch_size = actor_obs.shape[0]
        for i in range(batch_size):
            idx = self.ptr
            self.actor_obs[idx] = actor_obs[i]
            self.critic_obs[idx] = critic_obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.dones[idx] = dones[i]
            self.next_actor_obs[idx] = next_actor_obs[i]
            self.next_critic_obs[idx] = next_critic_obs[i]

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.actor_obs[idx],
            self.critic_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_actor_obs[idx],
            self.next_critic_obs[idx],
        )
