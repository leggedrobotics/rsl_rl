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
    """First transitions, raw reward windows and the last valid explicit successors."""

    observations: TensorDict
    actions: torch.Tensor
    window: dict[str | tuple[str, ...], torch.Tensor]
    valid: torch.Tensor
    next_observations: TensorDict
    terminated: torch.Tensor


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
        self._observations: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._next_observations: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._window: dict[str | tuple[str, ...], torch.Tensor] = {}
        self._actions: torch.Tensor | None = None

    @property
    def size(self) -> int:
        """Number of stored transitions across all environments."""
        return self.num_steps * self.num_envs

    @torch.no_grad()
    def add(self, step: TensorDict) -> None:
        """Copy observations, actions, rewards, successors, boolean dones/terminated and optional extra tensors."""
        if step.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected one transition per environment, got batch size {step.batch_size}.")
        leaves = dict(step.items(True, True))
        dones = leaves.get("dones")
        if dones is None or dones.shape != (self.num_envs,) or dones.dtype != torch.bool:
            raise ValueError("Replay transitions require boolean 'dones' with shape [num_envs].")
        if self.data is None:
            required = {"observations", "actions", "rewards", "next_observations", "dones", "terminated"}
            if not required.issubset(step.keys()):
                raise ValueError(f"Replay transitions require {sorted(required)}.")
            if step["terminated"].shape != (self.num_envs,) or step["terminated"].dtype != torch.bool:
                raise ValueError("Replay transitions require boolean 'terminated' with shape [num_envs].")
            if self.max_memory_fraction is not None and torch.device(self.device).type == "cuda":
                free, _ = torch.cuda.mem_get_info(torch.device(self.device))
                bytes_per_step = sum(value.numel() * value.element_size() for value in leaves.values())
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
                    for key, value in leaves.items()
                }
                self.data = TensorDict(self._data, batch_size=[self.num_slots, self.num_envs], device=self.device)
                for key, value in self._data.items():
                    flat = value.flatten(0, 1)
                    if isinstance(key, tuple) and key[0] in {"observations", "next_observations"}:
                        group = self._observations if key[0] == "observations" else self._next_observations
                        group[key[1] if len(key) == 2 else key[1:]] = flat
                    elif key == "actions":
                        self._actions = flat
                    else:
                        self._window[key] = flat
        if leaves.keys() != self._data.keys():
            raise ValueError("Replay transition keys changed after allocation.")
        for key, value in leaves.items():
            self._data[key][self.step].copy_(value)
        self.step = (self.step + 1) % self.num_slots
        self.num_steps = min(self.num_steps + 1, self.num_slots)

    def sample(self, batch_size: int, length: int = 1) -> ReplayBatch:
        """Sample raw windows, stopping at the first episode boundary or the newest stored transition."""
        if self.num_steps == 0:
            raise RuntimeError("Cannot sample empty replay.")
        if batch_size < 1 or length < 1:
            raise ValueError("Batch size and length must be positive.")
        anchor = torch.randint(self.num_steps, (batch_size,), device=self.device)
        env_idx = torch.randint(self.num_envs, (batch_size,), device=self.device)
        oldest = (self.step - self.num_steps) % self.num_slots
        if length == 1:
            first = last = ((oldest + anchor) % self.num_slots) * self.num_envs + env_idx
            ids = first[:, None]
            valid = torch.ones_like(ids, dtype=torch.bool)
        else:
            offsets = torch.arange(length, device=self.device)
            ages = anchor[:, None] + offsets
            valid = ages < self.num_steps
            ids = ((oldest + ages.clamp(max=self.num_steps - 1)) % self.num_slots) * self.num_envs + env_idx[:, None]
            dones = self._window["dones"][ids]
            alive = torch.cumprod((~dones[:, :-1]).long(), dim=1).bool()
            valid[:, 1:] &= alive
            last_offset = valid.sum(1, keepdim=True) - 1
            ids = ids.gather(1, torch.minimum(offsets, last_offset))
            first, last = ids[:, 0], ids[:, -1]
        return ReplayBatch(
            TensorDict(
                {key: value[first] for key, value in self._observations.items()}, [batch_size], device=self.device
            ),
            self._actions[first],
            {key: value[ids] for key, value in self._window.items()},
            valid,
            TensorDict(
                {key: value[last] for key, value in self._next_observations.items()}, [batch_size], device=self.device
            ),
            self._window["terminated"][last],
        )

    def clear(self) -> None:
        """Discard stored transitions while retaining allocated memory."""
        self.num_steps = self.step = 0
