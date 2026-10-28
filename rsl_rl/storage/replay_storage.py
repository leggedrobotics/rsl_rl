# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code ported from FlashSAC (flash_rl):
#   Copyright (c) 2026 Holiday Robotics
#   SPDX-License-Identifier: MIT
# The n-step return accumulation and uniform ring-buffer semantics are adapted
# from ``flash_rl/buffers/torch_buffer.py`` (TorchUniformBuffer), generalized to
# store TensorDict observation groups.

"""Uniform experience-replay storage for off-policy algorithms.

This is the off-policy counterpart to :class:`~rsl_rl.storage.RolloutStorage`.
It stores transitions in a fixed-capacity ring buffer and returns uniformly
sampled mini-batches. Observations are kept as :class:`~tensordict.TensorDict`
observation groups (matching the rest of RSL-RL), and n-step returns are
accumulated on insertion.
"""

from __future__ import annotations

import torch
from collections import deque
from dataclasses import dataclass
from tensordict import TensorDict


@dataclass
class _NStepStep:
    """One buffered vectorized transition awaiting n-step accumulation."""

    observation: TensorDict
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_observation: TensorDict


class ReplayStorage:
    """Fixed-capacity uniform replay buffer with n-step return accumulation.

    The buffer flattens the ``num_envs`` batch dimension into a single ring of
    ``capacity`` transitions. Each :meth:`add` call ingests one vectorized
    environment step (``num_envs`` transitions); once ``n_step`` steps are
    buffered, the earliest step is written with its n-step-accumulated reward,
    termination flags, and bootstrap observation.
    """

    class Batch:
        """A uniformly sampled mini-batch of transitions."""

        def __init__(
            self,
            observations: TensorDict,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            next_observations: TensorDict,
        ) -> None:
            """Store the sampled transition tensors."""
            self.observations = observations
            """Observations at the start of the transition."""
            self.actions = actions
            """Actions taken."""
            self.rewards = rewards
            """(N-step accumulated) rewards. Shape: (batch,)."""
            self.terminated = terminated
            """Terminal flags (true environment terminations). Shape: (batch,)."""
            self.truncated = truncated
            """Truncation flags (time-limit terminations). Shape: (batch,)."""
            self.next_observations = next_observations
            """Observations after the (n-step) transition, used for bootstrapping."""

    def __init__(
        self,
        num_envs: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        capacity: int,
        min_length: int,
        sample_batch_size: int,
        n_step: int,
        gamma: float,
        device: str = "cpu",
    ) -> None:
        """Allocate the ring buffers.

        Args:
            num_envs: Number of parallel environments per :meth:`add`.
            obs: A sample observation ``TensorDict`` used to infer group shapes/dtypes.
            actions_shape: Shape of a single action (without batch dimension).
            capacity: Maximum number of transitions stored.
            min_length: Minimum number of stored transitions before sampling is allowed.
            sample_batch_size: Number of transitions returned by :meth:`sample`.
            n_step: Number of steps for n-step return accumulation (``1`` = standard TD).
            gamma: Discount factor used for n-step accumulation.
            device: Device on which the buffers are allocated.
        """
        if capacity < num_envs:
            raise ValueError(f"Replay capacity ({capacity}) must be >= num_envs ({num_envs}).")
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}.")

        self.num_envs = num_envs
        self.capacity = capacity
        self.min_length = min_length
        self.sample_batch_size = sample_batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.device = device

        # Observation ring buffers (one per group), keyed like the environment obs.
        self.observations = TensorDict(
            {
                key: torch.zeros(capacity, *value.shape[1:], dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[capacity],
            device=device,
        )
        self.next_observations = TensorDict(
            {
                key: torch.zeros(capacity, *value.shape[1:], dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[capacity],
            device=device,
        )
        self.actions = torch.zeros(capacity, *actions_shape, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.terminated = torch.zeros(capacity, device=device)
        self.truncated = torch.zeros(capacity, device=device)

        # In-flight n-step window and ring cursors.
        self._n_step_window: deque[_NStepStep] = deque(maxlen=n_step)
        self._num_in_buffer = 0
        self._cursor = 0

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return self._num_in_buffer

    def add(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_observations: TensorDict,
    ) -> None:
        """Ingest one vectorized environment step and write an n-step transition when ready.

        All tensors carry a leading ``num_envs`` batch dimension. ``rewards``,
        ``terminated`` and ``truncated`` are shape ``(num_envs,)``.
        """
        self._n_step_window.append(
            _NStepStep(
                observation=observations.clone(),
                action=actions.detach().to(self.device).clone(),
                reward=rewards.detach().to(self.device).float().clone(),
                terminated=terminated.detach().to(self.device).float().clone(),
                truncated=truncated.detach().to(self.device).float().clone(),
                next_observation=next_observations.clone(),
            )
        )
        if len(self._n_step_window) < self.n_step:
            return

        obs_w, action_w, reward_w, terminated_w, truncated_w, next_obs_w = self._accumulate_n_step()
        self._write(obs_w, action_w, reward_w, terminated_w, truncated_w, next_obs_w)

    def _accumulate_n_step(
        self,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TensorDict]:
        """Compute the n-step transition for the earliest step in the window.

        Mirrors ``TorchUniformBuffer._get_n_step_prev_transition``: the reward is
        the discounted sum until the first ``done`` within the window, and the
        bootstrap observation / flags are taken from that first ``done`` (or the
        last step if none terminate).
        """
        first = self._n_step_window[0]
        last = self._n_step_window[-1]

        n_step_reward = last.reward.clone()
        n_step_terminated = last.terminated.clone()
        n_step_truncated = last.truncated.clone()
        n_step_next_obs = last.next_observation.clone()

        for idx in reversed(range(self.n_step - 1)):
            step = self._n_step_window[idx]
            reward = step.reward
            terminated = step.terminated
            truncated = step.truncated
            next_obs = step.next_observation

            done = torch.logical_or(terminated.bool(), truncated.bool()).float()
            n_step_reward = reward + self.gamma * n_step_reward * (1.0 - done)

            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            for key in n_step_next_obs.keys():  # noqa: SIM118 (TensorDict.keys() is the documented API)
                n_step_next_obs[key][done_mask] = next_obs[key][done_mask]

        return (
            first.observation,
            first.action,
            n_step_reward,
            n_step_terminated,
            n_step_truncated,
            n_step_next_obs,
        )

    def _write(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_observations: TensorDict,
    ) -> None:
        """Write ``num_envs`` transitions into the ring buffer, wrapping as needed."""
        add_batch = rewards.shape[0]
        end = self._cursor + add_batch
        if end <= self.capacity:
            idxs: slice | torch.Tensor = slice(self._cursor, end)
        else:
            idxs = (torch.arange(add_batch, device=self.device) + self._cursor) % self.capacity

        self.observations[idxs] = observations
        self.next_observations[idxs] = next_observations
        self.actions[idxs] = actions.to(self.actions.dtype)
        self.rewards[idxs] = rewards.to(self.rewards.dtype)
        self.terminated[idxs] = terminated.to(self.terminated.dtype)
        self.truncated[idxs] = truncated.to(self.truncated.dtype)

        self._num_in_buffer = min(self._num_in_buffer + add_batch, self.capacity)
        self._cursor = (self._cursor + add_batch) % self.capacity

    def can_sample(self) -> bool:
        """Return whether enough transitions have been stored to sample."""
        return self._num_in_buffer >= self.min_length

    def sample(self) -> Batch:
        """Return a uniformly sampled mini-batch of ``sample_batch_size`` transitions."""
        idxs = torch.randint(0, self._num_in_buffer, (self.sample_batch_size,), device=self.device)
        return ReplayStorage.Batch(
            observations=self.observations[idxs],  # type: ignore
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            terminated=self.terminated[idxs],
            truncated=self.truncated[idxs],
            next_observations=self.next_observations[idxs],  # type: ignore
        )

    def save(self, path: str) -> None:
        """Save the buffer contents and cursor metadata to ``path``."""
        n = self._num_in_buffer
        torch.save(
            {
                "observations": self.observations[:n],
                "next_observations": self.next_observations[:n],
                "actions": self.actions[:n],
                "rewards": self.rewards[:n],
                "terminated": self.terminated[:n],
                "truncated": self.truncated[:n],
                "num_in_buffer": self._num_in_buffer,
                "cursor": self._cursor,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load buffer contents saved by :meth:`save`.

        The in-flight n-step window is intentionally not persisted; at most
        ``n_step - 1`` transitions are lost, which is negligible.
        """
        data = torch.load(path, map_location=self.device, weights_only=False)
        n = data["num_in_buffer"]
        self.observations[:n] = data["observations"]
        self.next_observations[:n] = data["next_observations"]
        self.actions[:n] = data["actions"]
        self.rewards[:n] = data["rewards"]
        self.terminated[:n] = data["terminated"]
        self.truncated[:n] = data["truncated"]
        self._num_in_buffer = n
        self._cursor = data["cursor"]
        self._n_step_window.clear()
