# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from tensordict import TensorDict


@dataclass
class ReplayIndex:
    """Flat replay positions and the validity of each sampled transition."""

    ids: torch.Tensor
    valid: torch.Tensor


class SequenceReplayBuffer:
    """Lazy physical-transition replay with reset-aware observation history and future windows.

    Capacity counts transitions across environments and is rounded down to complete environment steps. Samples may
    include consecutive positions after an anchor; validity masks exclude missing positions and episode boundaries.
    The terminating transition itself remains valid. Invalid positions repeat the last valid position, so every
    gathered value comes from initialized storage. Storage is allocated on the first :meth:`add`.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        device: str = "cpu",
        min_steps: int = 1,
        max_memory_fraction: float | None = None,
    ) -> None:
        """Configure storage without allocating replay tensors."""
        if num_envs < 1 or capacity < num_envs or min_steps < 1:
            raise ValueError("Replay capacity must hold at least one step of all environments.")
        if max_memory_fraction is not None and not 0 < max_memory_fraction <= 1:
            raise ValueError("max_memory_fraction must lie in (0, 1].")
        self.num_envs = num_envs
        self.num_slots = capacity // num_envs
        self.device = device
        self.min_steps = min_steps
        self.max_memory_fraction = max_memory_fraction
        self.num_steps = 0
        self.step = 0
        self.data: TensorDict | None = None
        self._data: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._flat_data: dict[str | tuple[str, ...], torch.Tensor] = {}

    @property
    def size(self) -> int:
        """Number of stored transitions across all environments."""
        return self.num_steps * self.num_envs

    @torch.no_grad()
    def add(self, step: TensorDict) -> None:
        """Copy one environment step, including mandatory boolean ``dones`` flags."""
        if step.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected one transition per environment, got batch size {step.batch_size}.")
        if "dones" not in step or step["dones"].shape != (self.num_envs,) or step["dones"].dtype != torch.bool:
            raise ValueError("Replay transitions require boolean 'dones' with shape [num_envs].")
        if self.data is None:
            if self.max_memory_fraction is not None and torch.device(self.device).type == "cuda":
                free, _ = torch.cuda.mem_get_info(torch.device(self.device))
                bytes_per_step = sum(value.numel() * value.element_size() for value in step.values(True, True))
                affordable = int(self.max_memory_fraction * free) // bytes_per_step
                if affordable < self.num_slots:
                    warnings.warn(
                        f"Reducing replay from {self.num_slots} to {affordable} steps per environment to fit the"
                        " configured GPU memory fraction.",
                        stacklevel=2,
                    )
                    self.num_slots = affordable
            if self.num_slots < self.min_steps:
                raise ValueError(f"Replay capacity must hold at least {self.min_steps} steps per environment.")
            with torch.inference_mode(False):
                self._data = {
                    key: torch.empty(self.num_slots, *value.shape, dtype=value.dtype, device=self.device)
                    for key, value in step.items(True, True)
                }
                self.data = TensorDict(self._data, batch_size=[self.num_slots, self.num_envs], device=self.device)
                self._flat_data = {key: value.flatten(0, 1) for key, value in self._data.items()}
        if set(step.keys(True, True)) != self._data.keys():
            raise ValueError("Replay transition keys changed after allocation.")
        for key, value in step.items(True, True):
            self._data[key][self.step].copy_(value)
        self.step = (self.step + 1) % self.num_slots
        self.num_steps = min(self.num_steps + 1, self.num_slots)

    def sample(self, batch_size: int, length: int = 1, history: int = 0) -> ReplayIndex:
        """Sample positions of shape ``[batch_size, history+length]``, clipping history at resets and oldest storage.

        Unavailable history repeats the earliest valid position. Future positions repeat the last valid position.
        Validity remains explicit, including False on repeated padding positions.
        """
        if self.num_steps == 0:
            raise RuntimeError("Cannot sample empty replay.")
        if batch_size < 1 or length < 1 or history < 0:
            raise ValueError("Batch size and length must be positive; history must be nonnegative.")
        anchor = torch.randint(self.num_steps, (batch_size,), device=self.device)
        env_idx = torch.randint(self.num_envs, (batch_size,), device=self.device)
        ages = anchor[:, None] + torch.arange(length, device=self.device)
        stored = ages < self.num_steps
        oldest = (self.step - self.num_steps) % self.num_slots
        ids = ((oldest + ages.clamp(max=self.num_steps - 1)) % self.num_slots) * self.num_envs + env_idx[:, None]
        valid = stored
        if length > 1:
            dones = self._flat_data["dones"][ids]
            alive = torch.cumprod((~dones[:, :-1]).long(), dim=1).bool()
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

    def clear(self) -> None:
        """Discard stored transitions while retaining allocated memory."""
        self.num_steps = self.step = 0
