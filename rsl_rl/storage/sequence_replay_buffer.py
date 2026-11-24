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


@dataclass
class ReplayIndex:
    """Flat replay positions and the validity of each sampled transition."""

    ids: torch.Tensor
    valid: torch.Tensor


class SequenceReplayBuffer:
    """Store physical transitions and sample histories and windows that respect episode boundaries."""

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
        self.data: TensorDict | None = None
        self._flat_data: dict[str | tuple[str, ...], torch.Tensor] = {}

    @property
    def size(self) -> int:
        """Number of stored transitions across all environments."""
        return self.num_steps * self.num_envs

    @torch.no_grad()
    def add(self, step: TensorDict) -> None:
        """Copy one environment step, including mandatory boolean ``dones`` flags, and advance."""
        self.write(step)
        self.advance()

    @torch.no_grad()
    def write(self, step: TensorDict) -> None:
        """Copy one environment step into the current slot without advancing.

        The first call allocates storage. Subsequent writes use a device-side cursor and can be captured in CUDA graphs.
        """
        if step.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected one transition per environment, got batch size {step.batch_size}.")
        if "dones" not in step or step["dones"].shape != (self.num_envs,) or step["dones"].dtype != torch.bool:
            raise ValueError("Replay transitions require boolean 'dones' with shape [num_envs].")
        if self.data is None:
            self._initialize_storage(step)
        if set(step.keys(True, True)) != self._flat_data.keys():
            raise ValueError("Replay transition keys changed after allocation.")
        for key, value in step.items(True, True):
            stored = self.data[key]
            if value.shape != stored.shape[1:] or value.dtype != stored.dtype:
                raise ValueError(f"Replay transition shape or dtype changed for {key!r}.")

        slot = self._step_tensor.reshape(1)
        for key, value in step.items(True, True):
            self.data[key].index_copy_(0, slot, value.to(self.device).unsqueeze(0))

    def advance(self) -> None:
        """Move to the next slot after :meth:`write`."""
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

    def sample(self, batch_size: int, length: int = 1, history: int = 0) -> ReplayIndex:
        """Sample positions of shape ``[batch_size, history + length]`` within replay and episode boundaries.

        The terminating transition remains valid. History and future padding repeat the nearest valid position and
        have a false validity mask. External state edits cut future windows while histories still cross them.
        Sampling reads the device-side cursor and can be captured in CUDA graphs.
        """
        if self.num_steps == 0:
            raise RuntimeError("Cannot sample empty replay.")
        num_steps = self._num_steps_tensor
        anchor = (torch.rand(batch_size, device=self.device) * num_steps).long().clamp(max=num_steps - 1)
        env_idx = torch.randint(self.num_envs, (batch_size,), device=self.device)
        ages = anchor[:, None] + torch.arange(length, device=self.device)
        valid = ages < num_steps
        oldest = (self._step_tensor - num_steps) % self.num_slots
        ids = ((oldest + torch.minimum(ages, num_steps - 1)) % self.num_slots) * self.num_envs + env_idx[:, None]
        if length > 1:
            ends = self._flat_data["dones"][ids[:, :-1]]
            if "window_break" in self._flat_data:
                ends = ends | self._flat_data["window_break"][ids[:, :-1]]
            alive = torch.cumprod((~ends).long(), dim=1).bool()
            valid[:, 1:] &= alive
        last = valid.sum(1, keepdim=True) - 1
        ids = ids.gather(1, torch.minimum(torch.arange(length, device=self.device), last))
        if history:
            past_ages = anchor[:, None] - torch.arange(history, 0, -1, device=self.device)
            past_ids = ((oldest + past_ages.clamp(min=0)) % self.num_slots) * self.num_envs + env_idx[:, None]
            past_dones = self._flat_data["dones"][past_ids]
            past_valid = (past_ages >= 0) & torch.cumprod((~past_dones).flip(1).long(), dim=1).flip(1).bool()
            first = history - past_valid.sum(dim=1, keepdim=True)
            combined = torch.cat((past_ids, ids), dim=1)
            past_ids = combined.gather(1, torch.maximum(torch.arange(history, device=self.device), first))
            ids, valid = torch.cat((past_ids, ids), dim=1), torch.cat((past_valid, valid), dim=1)
        return ReplayIndex(ids, valid)

    def gather(self, keys: Sequence[str | tuple[str, ...]], ids: torch.Tensor) -> TensorDict:
        """Gather selected keys, preserving their dtype and the sampled batch/time dimensions."""
        if self.data is None:
            raise RuntimeError("Cannot gather empty replay.")
        selected = set(keys)
        return TensorDict(
            {
                key: value[ids]
                for key, value in self._flat_data.items()
                if key in selected or (isinstance(key, tuple) and key[0] in selected)
            },
            batch_size=ids.shape,
            device=self.device,
        )

    def complete_window_counts(self, length: int) -> torch.Tensor:
        """Return cumulative counts of valid starts ordered by environment and then age."""
        if length > self.num_slots:
            raise ValueError("Complete sequence length must fit replay capacity.")
        if self.num_steps < length:
            raise RuntimeError("Not enough stored states for a complete replay window.")
        ages = torch.arange(self.num_slots, device=self.device)
        oldest = (self._step_tensor - self._num_steps_tensor) % self.num_slots
        slots = (oldest + ages) % self.num_slots
        slots = torch.where(ages < self._num_steps_tensor, slots, oldest)
        ended = self.data["dones"][slots]
        if "window_break" in self.data:
            ended = ended | self.data["window_break"][slots]
        prefix = torch.cat((ended.new_zeros((1, self.num_envs)), ended), dim=0).long().cumsum(0)
        last = (ages + length - 1).clamp(max=self.num_slots)
        valid = (prefix[last] == prefix[ages]) & (ages[:, None] + length <= self._num_steps_tensor)
        return valid.T.flatten().cumsum(0)

    def sample_sequences(
        self, batch_size: int, length: int, cumulative_counts: torch.Tensor | None = None
    ) -> ReplayIndex:
        """Sample complete windows, uniformly over eligible environments and then their starts.

        Interior transitions cannot terminate or contain external state edits. Pass counts from
        :meth:`complete_window_counts` to reuse them across batches from the same replay contents.
        """
        if cumulative_counts is None:
            cumulative_counts = self.complete_window_counts(length)
            torch._assert_async(
                cumulative_counts[-1] > 0, "No complete replay windows remain within episode boundaries."
            )
        ends = cumulative_counts[self.num_slots - 1 :: self.num_slots]
        offsets = torch.nn.functional.pad(ends[:-1], (1, 0))
        counts = ends - offsets
        env_ids = torch.multinomial((counts > 0).float(), batch_size, replacement=True)
        choice = (torch.rand(batch_size, device=self.device) * counts[env_ids]).long()
        starts = torch.searchsorted(cumulative_counts, offsets[env_ids] + choice, right=True) % self.num_slots
        oldest = (self._step_tensor - self._num_steps_tensor) % self.num_slots
        slots = (oldest + starts[:, None] + torch.arange(length, device=self.device)) % self.num_slots
        ids = slots * self.num_envs + env_ids[:, None]
        return ReplayIndex(ids, torch.ones_like(ids, dtype=torch.bool))

    def _initialize_storage(self, step: TensorDict) -> None:
        """Allocate writable storage for the first transition's schema."""
        num_slots = _limit_replay_capacity(
            self.num_slots, self.max_memory_fraction, self.device, step.values(True, True)
        )
        if num_slots < self.min_steps_per_env:
            raise ValueError(f"Replay capacity must hold at least {self.min_steps_per_env} steps per environment.")
        with torch.inference_mode(False):
            data = {
                key: torch.empty(num_slots, *value.shape, dtype=value.dtype, device=self.device)
                for key, value in step.items(True, True)
            }
            self.data = TensorDict(data, batch_size=[num_slots, self.num_envs], device=self.device)
            self._flat_data = {key: value.flatten(0, 1) for key, value in data.items()}
        self.num_slots = num_slots
