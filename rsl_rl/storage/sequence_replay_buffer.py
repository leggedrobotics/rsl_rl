# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass
from tensordict import TensorDict

from rsl_rl.storage.replay_buffer import _limit_replay_capacity


class SequenceReplayBuffer:
    """Store physical transitions and sample histories and windows that respect episode boundaries."""

    @dataclass
    class BatchIndices:
        """Flat replay indices and the validity of sampled transitions."""

        indices: torch.Tensor
        valid: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        device: str = "cpu",
        min_steps_per_env: int = 1,
        max_memory_fraction: float | None = None,
    ) -> None:
        """Initialize the replay buffer without allocating transition tensors.

        Args:
            num_envs: Number of environments.
            capacity: Maximum number of transitions across all environments. Rounded down to complete environment steps.
            device: Device on which to store and sample transitions.
            min_steps_per_env: Minimum number of steps per environment that must fit in the buffer.
            max_memory_fraction: Maximum fraction of free CUDA memory available to replay. None disables the limit.
        """
        self.num_envs = num_envs
        self.num_slots = capacity // num_envs
        self.device = device
        self.min_steps_per_env = min_steps_per_env
        self.max_memory_fraction = max_memory_fraction

        self.num_steps = 0
        self.step = 0
        self._num_steps_tensor = torch.zeros((), dtype=torch.long, device=device)
        self._step_tensor = torch.zeros((), dtype=torch.long, device=device)
        self.transitions: TensorDict | None = None
        self._flat_transitions: dict[str | tuple[str, ...], torch.Tensor] = {}

    @property
    def size(self) -> int:
        """Number of stored transitions across all environments."""
        return self.num_steps * self.num_envs

    @torch.no_grad()
    def add_transition(self, transition: TensorDict) -> None:
        """Copy one environment step, including mandatory boolean ``dones`` flags, and advance."""
        self.write_transition(transition)
        self.advance()

    @torch.no_grad()
    def write_transition(self, transition: TensorDict) -> None:
        """Copy one environment step into the current slot without advancing.

        The first call allocates storage. Subsequent writes use a device-side cursor and can be captured in CUDA graphs.
        """
        if transition.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected one transition per environment, got batch size {transition.batch_size}.")
        dones = transition.get("dones", None)
        if dones is None or dones.shape != (self.num_envs,) or dones.dtype != torch.bool:
            raise ValueError("Replay transitions require boolean 'dones' with shape [num_envs].")

        if self.transitions is None:
            self._initialize_storage(transition)

        if set(transition.keys(include_nested=True, leaves_only=True)) != self._flat_transitions.keys():
            raise ValueError("Replay transition keys changed after allocation.")
        for key, value in transition.items(include_nested=True, leaves_only=True):
            stored = self.transitions[key]
            if value.shape != stored.shape[1:] or value.dtype != stored.dtype:
                raise ValueError(f"Replay transition shape or dtype changed for {key!r}.")

        slot = self._step_tensor.reshape(1)
        for key, value in transition.items(include_nested=True, leaves_only=True):
            self.transitions[key].index_copy_(0, slot, value.to(self.device).unsqueeze(0))

    def advance(self) -> None:
        """Move to the next slot after :meth:`write_transition`."""
        self.step = (self.step + 1) % self.num_slots
        self.num_steps = min(self.num_steps + 1, self.num_slots)
        self._step_tensor.fill_(self.step)
        self._num_steps_tensor.fill_(self.num_steps)

    def clear(self) -> None:
        """Discard stored transitions while retaining allocated memory."""
        self.num_steps = 0
        self.step = 0
        self._num_steps_tensor.zero_()
        self._step_tensor.zero_()

    def sample(self, batch_size: int, length: int = 1, history_length: int = 0) -> SequenceReplayBuffer.BatchIndices:
        """Sample indices of shape ``[batch_size, history_length + length]`` within replay and episode boundaries.

        The terminating transition remains valid. History and future padding repeat the nearest valid position and
        have a false validity mask. External state edits cut future windows while histories still cross them.
        Sampling reads the device-side cursor and can be captured in CUDA graphs.
        """
        if self.num_steps == 0:
            raise RuntimeError("Cannot sample empty replay.")

        num_steps = self._num_steps_tensor
        anchor_steps = (torch.rand(batch_size, device=self.device) * num_steps).long().clamp(max=num_steps - 1)
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device)
        oldest_slot = (self._step_tensor - num_steps) % self.num_slots

        offsets = torch.arange(length, device=self.device)
        ages = anchor_steps[:, None] + offsets
        valid = ages < num_steps
        slots = (oldest_slot + torch.minimum(ages, num_steps - 1)) % self.num_slots
        indices = slots * self.num_envs + env_ids[:, None]

        if length > 1:
            window_ends = self._flat_transitions["dones"][indices[:, :-1]]
            if "window_break" in self._flat_transitions:
                window_ends = window_ends | self._flat_transitions["window_break"][indices[:, :-1]]
            valid[:, 1:] &= torch.cumprod((~window_ends).long(), dim=1).bool()
        last_valid_offset = valid.sum(1, keepdim=True) - 1
        indices = indices.gather(1, torch.minimum(offsets, last_valid_offset))

        if history_length:
            history_ages = anchor_steps[:, None] - torch.arange(history_length, 0, -1, device=self.device)
            history_slots = (oldest_slot + history_ages.clamp(min=0)) % self.num_slots
            history_indices = history_slots * self.num_envs + env_ids[:, None]
            history_dones = self._flat_transitions["dones"][history_indices]
            history_valid = (history_ages >= 0) & torch.cumprod((~history_dones).flip(1).long(), dim=1).flip(1).bool()

            first_valid_offset = history_length - history_valid.sum(dim=1, keepdim=True)
            history_offsets = torch.arange(history_length, device=self.device)
            combined_indices = torch.cat((history_indices, indices), dim=1)
            history_indices = combined_indices.gather(1, torch.maximum(history_offsets, first_valid_offset))
            indices = torch.cat((history_indices, indices), dim=1)
            valid = torch.cat((history_valid, valid), dim=1)

        return SequenceReplayBuffer.BatchIndices(indices, valid)

    def compute_complete_window_counts(self, length: int) -> torch.Tensor:
        """Return cumulative counts of valid starts ordered by environment and then age."""
        if length > self.num_slots:
            raise ValueError("Complete sequence length must fit replay capacity.")
        if self.num_steps < length:
            raise RuntimeError("Not enough stored states for a complete replay window.")

        ages = torch.arange(self.num_slots, device=self.device)
        oldest_slot = (self._step_tensor - self._num_steps_tensor) % self.num_slots
        slots = (oldest_slot + ages) % self.num_slots
        slots = torch.where(ages < self._num_steps_tensor, slots, oldest_slot)
        window_ends = self.transitions["dones"][slots]
        if "window_break" in self.transitions:
            window_ends = window_ends | self.transitions["window_break"][slots]

        cumulative_ends = torch.cat((window_ends.new_zeros((1, self.num_envs)), window_ends)).long().cumsum(0)
        last_age = (ages + length - 1).clamp(max=self.num_slots)
        valid_starts = (cumulative_ends[last_age] == cumulative_ends[ages]) & (
            ages[:, None] + length <= self._num_steps_tensor
        )
        return valid_starts.T.flatten().cumsum(0)

    def sample_complete_windows(
        self, batch_size: int, length: int, cumulative_counts: torch.Tensor | None = None
    ) -> SequenceReplayBuffer.BatchIndices:
        """Sample complete windows, uniformly over eligible environments and then their starts.

        Interior transitions cannot terminate or contain external state edits. Pass counts from
        :meth:`compute_complete_window_counts` to reuse them across batches from the same replay contents.
        """
        if cumulative_counts is None:
            cumulative_counts = self.compute_complete_window_counts(length)
            torch._assert_async(
                cumulative_counts[-1] > 0, "No complete replay windows remain within episode boundaries."
            )

        env_totals = cumulative_counts[self.num_slots - 1 :: self.num_slots]
        env_offsets = torch.nn.functional.pad(env_totals[:-1], (1, 0))
        env_counts = env_totals - env_offsets
        env_ids = torch.multinomial((env_counts > 0).float(), batch_size, replacement=True)
        start_offsets = (torch.rand(batch_size, device=self.device) * env_counts[env_ids]).long()
        start_ages = torch.searchsorted(cumulative_counts, env_offsets[env_ids] + start_offsets, right=True)

        oldest_slot = (self._step_tensor - self._num_steps_tensor) % self.num_slots
        offsets = start_ages[:, None] % self.num_slots + torch.arange(length, device=self.device)
        slots = (oldest_slot + offsets) % self.num_slots
        indices = slots * self.num_envs + env_ids[:, None]
        return SequenceReplayBuffer.BatchIndices(indices, torch.ones_like(indices, dtype=torch.bool))

    def gather_transitions(self, keys: Sequence[str | tuple[str, ...]], indices: torch.Tensor) -> TensorDict:
        """Gather selected keys, preserving their dtype and the sampled batch/time dimensions."""
        if self.transitions is None:
            raise RuntimeError("Cannot gather empty replay.")
        selected = set(keys)
        return TensorDict(
            {
                key: value[indices]
                for key, value in self._flat_transitions.items()
                if key in selected or (isinstance(key, tuple) and key[0] in selected)
            },
            batch_size=indices.shape,
            device=self.device,
        )

    def _initialize_storage(self, transition: TensorDict) -> None:
        """Allocate writable storage for the first transition's schema."""
        num_slots = _limit_replay_capacity(
            self.num_slots,
            self.max_memory_fraction,
            self.device,
            transition.values(include_nested=True, leaves_only=True),
        )
        if num_slots < self.min_steps_per_env:
            raise ValueError(f"Replay capacity must hold at least {self.min_steps_per_env} steps per environment.")

        with torch.inference_mode(False):
            transition_tensors = {
                key: torch.empty(num_slots, *value.shape, dtype=value.dtype, device=self.device)
                for key, value in transition.items(include_nested=True, leaves_only=True)
            }
            self.transitions = TensorDict(transition_tensors, batch_size=[num_slots, self.num_envs], device=self.device)
            self._flat_transitions = {key: value.flatten(0, 1) for key, value in transition_tensors.items()}
        self.num_slots = num_slots
