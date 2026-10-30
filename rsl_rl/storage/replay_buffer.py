# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import warnings
from dataclasses import dataclass
from tensordict import TensorDict


@dataclass
class ReplayBatch:
    """A batch of replay transitions and chronological reward windows.

    Observations and actions come from the first transition in each window. Next observations and termination flags
    come from the last valid transition. The validity mask excludes transitions after episode boundaries or beyond
    the newest stored step.
    """

    observations: TensorDict
    """Observations at the first transition of each window."""

    actions: torch.Tensor
    """Actions at the first transition of each window."""

    window: dict[str | tuple[str, ...], torch.Tensor]
    """Raw rewards, episode flags, and extra fields over the sampled windows."""

    valid: torch.Tensor
    """Validity masks for window entries. Shape: (batch_size, length)."""

    next_observations: TensorDict
    """Observations after the last valid transition of each window."""

    terminated: torch.Tensor
    """Termination flags at the last valid transition of each window."""


class ReplayBuffer:
    """Lazy dense replay for SAC, with chronological per-environment transition windows.

    Capacity counts transitions across environments and is rounded down to complete environment steps. Samples may
    include consecutive positions after an anchor; validity masks exclude missing positions and episode boundaries.
    The terminating transition itself remains valid. Invalid positions repeat the last valid position, so every
    gathered value comes from initialized storage. Storage is allocated on the first :meth:`add`.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        min_steps: int = 1,
        max_memory_fraction: float | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the replay buffer without allocating transition tensors.

        Args:
            num_envs: Number of environments.
            capacity: Maximum number of transitions across all environments. Rounded down to complete environment
                steps.
            min_steps: Minimum number of steps per environment that must fit in the buffer.
            max_memory_fraction: Maximum fraction of free GPU memory available for replay allocation. ``None``
                disables the memory limit. Only applies to CUDA devices.
            device: Device on which to store and sample transitions.
        """
        # Check the replay capacity and memory limit
        if num_envs < 1 or capacity < num_envs or min_steps < 1:
            raise ValueError("Replay capacity must hold at least one step of all environments.")
        if max_memory_fraction is not None and not 0 < max_memory_fraction <= 1:
            raise ValueError("max_memory_fraction must lie in (0, 1].")

        # Replay parameters
        self.num_envs = num_envs
        self.num_slots = capacity // num_envs
        self.device = device
        self.min_steps = min_steps
        self.max_memory_fraction = max_memory_fraction

        # Initialize the write cursor and allocation state
        self.num_steps = 0
        self.step = 0
        self.data: TensorDict | None = None
        self._data: dict[str | tuple[str, ...], torch.Tensor] = {}

        # Cache flattened views for sampling observations, actions, and reward windows
        self._observations: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._next_observations: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._window: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._actions: torch.Tensor | None = None

    @property
    def size(self) -> int:
        """Return the number of stored transitions across all environments."""
        return self.num_steps * self.num_envs

    @torch.no_grad()
    def add(self, step: TensorDict) -> None:
        """Copy one transition per environment into the replay buffer.

        Storage is allocated on the first call using the shapes and dtypes of `step`.

        Args:
            step: Transition dictionary with observations, actions, rewards, next observations, done flags, and
                termination flags. Done and termination flags must be boolean tensors of shape ``(num_envs,)``.
                Additional tensor fields are stored alongside the required fields.
        """
        # Check the transition batch shape and done flags
        if step.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected one transition per environment, got batch size {step.batch_size}.")
        leaves = dict(step.items(True, True))
        dones = leaves.get("dones")
        if dones is None or dones.shape != (self.num_envs,) or dones.dtype != torch.bool:
            raise ValueError("Replay transitions require boolean 'dones' with shape [num_envs].")

        # Initialize storage from the first transition
        if self.data is None:
            # Check the required fields and termination flags
            required = {"observations", "actions", "rewards", "next_observations", "dones", "terminated"}
            if not required.issubset(step.keys()):
                raise ValueError(f"Replay transitions require {sorted(required)}.")
            if step["terminated"].shape != (self.num_envs,) or step["terminated"].dtype != torch.bool:
                raise ValueError("Replay transitions require boolean 'terminated' with shape [num_envs].")

            # Limit capacity to the configured fraction of free GPU memory
            if self.max_memory_fraction is not None and torch.device(self.device).type == "cuda":
                free_memory, _ = torch.cuda.mem_get_info(torch.device(self.device))
                bytes_per_step = sum(value.numel() * value.element_size() for value in leaves.values())
                num_affordable_steps = int(self.max_memory_fraction * free_memory) // bytes_per_step
                if num_affordable_steps < self.num_slots:
                    warnings.warn(
                        f"Reducing replay from {self.num_slots} to {num_affordable_steps} steps per environment"
                        " to fit the configured GPU memory fraction.",
                        stacklevel=2,
                    )
                    self.num_slots = num_affordable_steps

            # Ensure the minimum replay window still fits
            if self.num_slots < self.min_steps:
                raise ValueError(f"Replay capacity must hold at least {self.min_steps} steps per environment.")

            # Allocate mutable tensors even when collection runs under inference mode
            with torch.inference_mode(False):
                self._data = {
                    key: torch.empty(self.num_slots, *value.shape, dtype=value.dtype, device=self.device)
                    for key, value in leaves.items()
                }
                self.data = TensorDict(self._data, batch_size=[self.num_slots, self.num_envs], device=self.device)

                # Cache flattened views of each transition field for indexed sampling
                for key, value in self._data.items():
                    flat_value = value.flatten(0, 1)
                    if isinstance(key, tuple) and key[0] in {"observations", "next_observations"}:
                        group = self._observations if key[0] == "observations" else self._next_observations
                        group[key[1] if len(key) == 2 else key[1:]] = flat_value
                    elif key == "actions":
                        self._actions = flat_value
                    else:
                        self._window[key] = flat_value

        # Copy the transition into the ring buffer and advance the write cursor
        if leaves.keys() != self._data.keys():
            raise ValueError("Replay transition keys changed after allocation.")
        for key, value in leaves.items():
            self._data[key][self.step].copy_(value)
        self.step = (self.step + 1) % self.num_slots
        self.num_steps = min(self.num_steps + 1, self.num_slots)

    def sample(self, batch_size: int, length: int = 1) -> ReplayBatch:
        """Sample windows of consecutive transitions for independently selected environments.

        Windows stop at the first episode boundary or the newest stored transition. Invalid positions repeat the
        last valid transition and are excluded by :attr:`ReplayBatch.valid`.

        Args:
            batch_size: Number of windows to sample.
            length: Maximum number of consecutive transitions in each window.

        Returns:
            The sampled transitions, raw reward windows, and validity masks.
        """
        if self.num_steps == 0:
            raise RuntimeError("Cannot sample empty replay.")
        if batch_size < 1 or length < 1:
            raise ValueError("Batch size and length must be positive.")

        # Sample anchor steps and environments independently
        anchor = torch.randint(self.num_steps, (batch_size,), device=self.device)
        env_idx = torch.randint(self.num_envs, (batch_size,), device=self.device)
        oldest = (self.step - self.num_steps) % self.num_slots

        # Select window entries in chronological order
        if length == 1:
            first = ((oldest + anchor) % self.num_slots) * self.num_envs + env_idx
            last = first
            indices = first[:, None]
            valid = torch.ones_like(indices, dtype=torch.bool)
        else:
            offsets = torch.arange(length, device=self.device)
            ages = anchor[:, None] + offsets
            valid = ages < self.num_steps
            step_indices = (oldest + ages.clamp(max=self.num_steps - 1)) % self.num_slots
            indices = step_indices * self.num_envs + env_idx[:, None]

            # Stop at episode boundaries and repeat the last valid transition at invalid positions
            dones = self._window["dones"][indices]
            alive = torch.cumprod((~dones[:, :-1]).long(), dim=1).bool()
            valid[:, 1:] &= alive
            last_offset = valid.sum(1, keepdim=True) - 1
            indices = indices.gather(1, torch.minimum(offsets, last_offset))
            first = indices[:, 0]
            last = indices[:, -1]

        # Construct the batch using the first and last valid transitions of each window
        return ReplayBatch(
            observations=TensorDict(
                {key: value[first] for key, value in self._observations.items()},
                batch_size=[batch_size],
                device=self.device,
            ),
            actions=self._actions[first],
            window={key: value[indices] for key, value in self._window.items()},
            valid=valid,
            next_observations=TensorDict(
                {key: value[last] for key, value in self._next_observations.items()},
                batch_size=[batch_size],
                device=self.device,
            ),
            terminated=self._window["terminated"][last],
        )

    def clear(self) -> None:
        """Discard stored transitions while retaining allocated memory."""
        self.num_steps = 0
        self.step = 0
