# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import warnings
from dataclasses import dataclass
from tensordict import TensorDict


class ReplayBuffer:
    """Storage for transitions collected from the environment.

    The replay buffer stores past transitions and returns random batches for learning. New transitions replace the
    oldest ones when the buffer is full.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated during collection and passed to :meth:`ReplayBuffer.add_transition`.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            """Observations at the current step."""

            self.actions: torch.Tensor | None = None
            """Actions taken at the current step."""

            self.rewards: torch.Tensor | None = None
            """Rewards received after the action."""

            self.next_observations: TensorDict | None = None
            """Observations after the action, before any environment reset."""

            self.dones: torch.Tensor | None = None
            """Boolean flags indicating episode boundaries. Shape: (num_envs,)."""

            self.terminated: torch.Tensor | None = None
            """Boolean flags indicating terminal states, excluding timeouts. Shape: (num_envs,)."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    @dataclass
    class Batch:
        """A batch of data returned by :meth:`ReplayBuffer.sample`.

        Each batch entry contains one or more consecutive transitions from the same environment. Observations and
        actions are taken from the first step. Next observations and termination flags are taken from the last valid
        step. Rewards and done flags are included for every step, with a mask marking the valid entries.
        """

        observations: TensorDict
        """Observations at the first transition of each window."""

        actions: torch.Tensor
        """Actions at the first transition of each window."""

        window: dict[str, torch.Tensor]
        """Raw rewards and episode flags over the sampled windows."""

        valid: torch.Tensor
        """Validity masks for window entries. Shape: (batch_size, length)."""

        next_observations: TensorDict
        """Observations after the last valid transition of each window."""

        terminated: torch.Tensor
        """Termination flags at the last valid transition of each window."""

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        min_steps_per_env: int = 1,
        max_memory_fraction: float | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the replay buffer without allocating transition tensors.

        Args:
            num_envs: Number of environments.
            capacity: Maximum number of transitions across all environments. Rounded down to complete environment steps.
            min_steps_per_env: Minimum number of transitions per environment that must fit in the buffer.
            max_memory_fraction: Maximum fraction of free GPU memory available for replay allocation. ``None`` disables
                the memory limit. Only applies to CUDA devices.
            device: Device on which to store and sample transitions.
        """
        # Replay parameters
        self.num_envs = num_envs
        self.num_slots = capacity // num_envs
        self.min_steps_per_env = min_steps_per_env
        self.max_memory_fraction = max_memory_fraction
        self.device = device

        # Initialize the write cursor and storage fields
        self.num_steps = 0
        self.step = 0
        self.observations: TensorDict | None = None
        self.actions: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.next_observations: TensorDict | None = None
        self.dones: torch.Tensor | None = None
        self.terminated: torch.Tensor | None = None

    @property
    def size(self) -> int:
        """Return the number of stored transitions across all environments."""
        return self.num_steps * self.num_envs

    def _initialize_storage(self, transition: ReplayBuffer.Transition) -> None:
        """Allocate replay tensors using the shapes and dtypes of the first transition."""
        # Limit capacity to the configured fraction of free GPU memory
        if self.max_memory_fraction is not None and torch.device(self.device).type == "cuda":
            free_memory, _ = torch.cuda.mem_get_info(torch.device(self.device))
            bytes_per_step = sum(
                value.numel() * value.element_size()  # type: ignore
                for value in (
                    *transition.observations.values(include_nested=True, leaves_only=True),
                    transition.actions,
                    transition.rewards,
                    *transition.next_observations.values(include_nested=True, leaves_only=True),
                    transition.dones,
                    transition.terminated,
                )
            )
            num_affordable_steps = int(self.max_memory_fraction * free_memory) // bytes_per_step
            if num_affordable_steps < self.num_slots:
                warnings.warn(
                    f"Reducing replay from {self.num_slots} to {num_affordable_steps} steps per environment"
                    " to fit the configured GPU memory fraction.",
                )
                self.num_slots = num_affordable_steps

        # Ensure the minimum replay window still fits
        if self.num_slots < self.min_steps_per_env:
            raise ValueError(f"Replay capacity must hold at least {self.min_steps_per_env} steps per environment.")

        # Allocate mutable tensors even when collection runs under inference mode
        with torch.inference_mode(False):
            self.observations = TensorDict(
                {
                    key: torch.empty(self.num_slots, *value.shape, dtype=value.dtype, device=self.device)
                    for key, value in transition.observations.items(include_nested=True, leaves_only=True)
                },
                batch_size=[self.num_slots, self.num_envs],
                device=self.device,
            )
            self.actions = torch.empty(
                self.num_slots, *transition.actions.shape, dtype=transition.actions.dtype, device=self.device
            )
            self.rewards = torch.empty(
                self.num_slots, *transition.rewards.shape, dtype=transition.rewards.dtype, device=self.device
            )
            self.next_observations = TensorDict(
                {
                    key: torch.empty(self.num_slots, *value.shape, dtype=value.dtype, device=self.device)
                    for key, value in transition.next_observations.items(include_nested=True, leaves_only=True)
                },
                batch_size=[self.num_slots, self.num_envs],
                device=self.device,
            )
            self.dones = torch.empty(self.num_slots, self.num_envs, dtype=torch.bool, device=self.device)
            self.terminated = torch.empty(self.num_slots, self.num_envs, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def add_transition(self, transition: ReplayBuffer.Transition) -> None:
        """Copy one transition per environment into the replay buffer.

        Storage is allocated on the first call using the shapes and dtypes of `transition`.
        """
        # Initialize storage
        if self.observations is None:
            self._initialize_storage(transition)

        # Copy the transition into the ring buffer and advance the write cursor
        self.observations[self.step].copy_(transition.observations)  # type: ignore
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards)  # type: ignore
        self.next_observations[self.step].copy_(transition.next_observations)  # type: ignore
        self.dones[self.step].copy_(transition.dones)  # type: ignore
        self.terminated[self.step].copy_(transition.terminated)  # type: ignore
        self.step = (self.step + 1) % self.num_slots
        self.num_steps = min(self.num_steps + 1, self.num_slots)

    def sample(self, batch_size: int, length: int = 1) -> ReplayBuffer.Batch:
        """Sample windows of consecutive transitions for independently selected environments.

        Windows stop at the first episode boundary or the newest stored transition. Invalid positions repeat the
        last valid transition and are excluded by :attr:`ReplayBuffer.Batch.valid`.

        Args:
            batch_size: Number of windows to sample.
            length: Maximum number of consecutive transitions in each window.
        """
        if self.num_steps == 0:
            raise RuntimeError("Cannot sample empty replay.")

        # Sample anchor steps and environments independently
        anchor = torch.randint(self.num_steps, (batch_size,), device=self.device)
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device)
        oldest = (self.step - self.num_steps) % self.num_slots

        # Compute the indices of sampled windows and their validity masks
        if length == 1:
            indices = ((oldest + anchor) % self.num_slots)[:, None]
            valid = torch.ones_like(indices, dtype=torch.bool)
        else:
            anchor_offsets = torch.arange(length, device=self.device)
            offsets = anchor[:, None] + anchor_offsets
            valid = offsets < self.num_steps
            indices = (oldest + offsets.clamp(max=self.num_steps - 1)) % self.num_slots

            # Stop at episode boundaries and repeat the last valid transition at invalid positions
            dones = self.dones[indices, env_ids[:, None]]  # type: ignore
            alive = torch.cumprod((~dones[:, :-1]).long(), dim=1).bool()
            valid[:, 1:] &= alive
            last_valid_offset = valid.sum(1, keepdim=True) - 1
            indices = indices.gather(1, torch.minimum(anchor_offsets, last_valid_offset))
        first = indices[:, 0]
        last = indices[:, -1]

        # Construct the batch using the first and last valid transitions of each window
        return ReplayBuffer.Batch(
            observations=self.observations[first, env_ids],  # type: ignore
            actions=self.actions[first, env_ids],  # type: ignore
            window={
                "rewards": self.rewards[indices, env_ids[:, None]],  # type: ignore
                "dones": self.dones[indices, env_ids[:, None]],  # type: ignore
                "terminated": self.terminated[indices, env_ids[:, None]],  # type: ignore
            },
            valid=valid,
            next_observations=self.next_observations[last, env_ids],  # type: ignore
            terminated=self.terminated[last, env_ids],  # type: ignore
        )

    def clear(self) -> None:
        """Discard stored transitions while retaining allocated memory."""
        self.num_steps = 0
        self.step = 0
