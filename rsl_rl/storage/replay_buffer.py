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
        step. Rewards are included for every step, with a mask marking the valid entries.
        """

        observations: TensorDict
        """Observations at the first transition of each window."""

        actions: torch.Tensor
        """Actions at the first transition of each window."""

        rewards: torch.Tensor
        """Raw rewards over the sampled windows. Shape: (batch_size, length)."""

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
        self._num_steps_tensor = torch.zeros((), dtype=torch.long, device=device)
        self._step_tensor = torch.zeros((), dtype=torch.long, device=device)
        self.transitions: TensorDict | None = None
        self._flat_transitions: TensorDict | None = None
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

    @property
    def write_index(self) -> torch.Tensor:
        """Device tensor with the slot that the next :meth:`write_transition` writes to."""
        return self._step_tensor

    @torch.no_grad()
    def add_transition(self, transition: ReplayBuffer.Transition | TensorDict) -> None:
        """Copy one transition per environment into the replay buffer.

        Pass a :class:`Transition` or a TensorDict of per-environment fields with boolean ``dones`` flags.
        An optional boolean ``window_break`` flag cuts forward windows at external state edits.
        The first transition defines the stored fields, shapes and dtypes.
        """
        self.write_transition(transition)
        self.advance()

    @torch.no_grad()
    def write_transition(self, transition: ReplayBuffer.Transition | TensorDict) -> None:
        """Copy one transition into the slot at the device-side write cursor and advance that cursor.

        After the first call, which allocates the storage, this method only runs device operations, so it can be
        captured in a CUDA graph. Each call must be followed by :meth:`advance`.
        """
        if isinstance(transition, ReplayBuffer.Transition):
            transition = TensorDict(vars(transition), batch_size=[self.num_envs])

        if transition.batch_size != torch.Size([self.num_envs]):
            raise ValueError(f"Expected one transition per environment, got batch size {transition.batch_size}.")
        dones = transition.get("dones", None)
        if dones is None or dones.shape != (self.num_envs,) or dones.dtype != torch.bool:
            raise ValueError("Replay transitions require boolean 'dones' with shape [num_envs].")

        if self.transitions is None:
            self._initialize_storage(transition)

        if set(transition.keys(True, True)) != set(self.transitions.keys(True, True)):
            raise ValueError("Replay transition keys changed after allocation.")
        for key, value in transition.items(include_nested=True, leaves_only=True):
            stored = self.transitions[key]
            if value.shape != stored.shape[1:] or value.dtype != stored.dtype:
                raise ValueError(f"Replay transition shape or dtype changed for {key!r}.")

        slot = self._step_tensor.reshape(1)
        for key, value in transition.items(include_nested=True, leaves_only=True):
            self.transitions[key].index_copy_(0, slot, value.to(self.device).unsqueeze(0))

        # Advance the device-side write cursor
        self._step_tensor.copy_((self._step_tensor + 1) % self.num_slots)
        self._num_steps_tensor.copy_((self._num_steps_tensor + 1).clamp(max=self.num_slots))

    def advance(self) -> None:
        """Advance the host-side write cursor after :meth:`write_transition`."""
        self.step = (self.step + 1) % self.num_slots
        self.num_steps = min(self.num_steps + 1, self.num_slots)

    def clear(self) -> None:
        """Discard stored transitions while retaining allocated memory."""
        self.num_steps = 0
        self.step = 0
        self._num_steps_tensor.zero_()
        self._step_tensor.zero_()

    def latest_indices(self, num_steps: int) -> torch.Tensor:
        """Return the slots of the latest ``num_steps`` transitions, oldest first."""
        num_steps = min(num_steps, self.num_steps)
        return torch.arange(self.step - num_steps, self.step, device=self.device) % self.num_slots

    def latest_observations(self, num_steps: int) -> TensorDict:
        """Return the observations of the latest ``num_steps`` transitions, flattened over steps and environments."""
        return self.observations[self.latest_indices(num_steps)].reshape(-1)  # type: ignore

    def sample(self, batch_size: int, length: int = 1) -> ReplayBuffer.Batch:
        """Sample windows of consecutive transitions for independently selected environments.

        Windows stop at the first episode boundary or the newest stored transition. Invalid positions repeat the
        last valid transition and are excluded by :attr:`ReplayBuffer.Batch.valid`. Sampling reads the fill level from
        device tensors, so it can be captured in a CUDA graph.

        Args:
            batch_size: Number of windows to sample.
            length: Maximum number of consecutive transitions in each window.
        """
        indices, valid = self._sample_indices(batch_size, length)
        first = indices[:, 0]
        last = indices[:, -1]
        flat = self._flat_transitions
        return ReplayBuffer.Batch(
            observations=flat["observations"][first],
            actions=flat["actions"][first],
            rewards=flat["rewards"][indices],
            valid=valid,
            next_observations=flat["next_observations"][last],
            terminated=flat["terminated"][last],
        )

    def sample_sequence(
        self,
        batch_size: int,
        keys: Sequence[str | tuple[str, ...]],
        length: int = 1,
        history_length: int = 0,
        complete_window_counts: torch.Tensor | None = None,
    ) -> tuple[TensorDict, torch.Tensor]:
        """Return selected fields and their validity over a history and forward window.

        History stops at resets. Forward windows also stop at external state edits flagged by ``window_break``.
        Padding repeats the nearest valid transition. Pass counts from :meth:`compute_complete_window_counts`
        from the current replay contents to sample eligible environments and then their starts uniformly. The
        length used to compute the counts may include extra lookahead beyond the requested window.
        """
        if complete_window_counts is None:
            indices, valid = self._sample_indices(batch_size, length, history_length)
        else:
            if history_length:
                raise ValueError("Complete forward windows cannot include padded history.")
            indices = self._sample_complete_windows(batch_size, length, complete_window_counts)
            valid = torch.ones_like(indices, dtype=torch.bool)
        return self._flat_transitions.select(*keys)[indices], valid

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

    def _sample_indices(
        self, batch_size: int, length: int = 1, history_length: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        if length == 1:
            indices = (((oldest_slot + anchor_steps) % self.num_slots) * self.num_envs + env_ids)[:, None]
            valid = torch.ones_like(indices, dtype=torch.bool)
        else:
            offsets = torch.arange(length, device=self.device)
            ages = anchor_steps[:, None] + offsets
            valid = ages < num_steps
            slots = (oldest_slot + torch.minimum(ages, num_steps - 1)) % self.num_slots
            indices = slots * self.num_envs + env_ids[:, None]

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

        return indices, valid

    def _sample_complete_windows(self, batch_size: int, length: int, cumulative_counts: torch.Tensor) -> torch.Tensor:
        """Sample complete windows, uniformly over eligible environments and then their starts.

        Interior transitions cannot terminate or contain external state edits. Pass counts from
        :meth:`compute_complete_window_counts` to reuse them across batches from the same replay contents.
        """
        env_totals = cumulative_counts[self.num_slots - 1 :: self.num_slots]
        env_offsets = torch.nn.functional.pad(env_totals[:-1], (1, 0))
        env_counts = env_totals - env_offsets
        env_ids = torch.multinomial((env_counts > 0).float(), batch_size, replacement=True)
        start_offsets = (torch.rand(batch_size, device=self.device) * env_counts[env_ids]).long()
        start_ages = torch.searchsorted(cumulative_counts, env_offsets[env_ids] + start_offsets, right=True)

        oldest_slot = (self._step_tensor - self._num_steps_tensor) % self.num_slots
        offsets = start_ages[:, None] % self.num_slots + torch.arange(length, device=self.device)
        slots = (oldest_slot + offsets) % self.num_slots
        return slots * self.num_envs + env_ids[:, None]

    def _initialize_storage(self, transition: TensorDict) -> None:
        """Allocate writable storage for the first transition's schema."""
        num_slots = self.num_slots
        if self.max_memory_fraction is not None and torch.device(self.device).type == "cuda":
            free_memory, _ = torch.cuda.mem_get_info(torch.device(self.device))
            bytes_per_step = sum(t.numel() * t.element_size() for t in transition.values(True, True))
            affordable = int(self.max_memory_fraction * free_memory) // bytes_per_step
            if affordable < num_slots:
                warnings.warn(
                    f"Reducing replay from {num_slots} to {affordable} steps per environment"
                    " to fit the configured GPU memory fraction.",
                    stacklevel=3,
                )
                num_slots = affordable
        if num_slots < self.min_steps_per_env:
            raise ValueError(f"Replay capacity must hold at least {self.min_steps_per_env} steps per environment.")

        with torch.inference_mode(False):
            self.transitions = transition.new_empty(num_slots, self.num_envs, device=self.device)
            self._flat_transitions = self.transitions.view(-1)
            self.observations = self.transitions.get("observations", None)
            self.next_observations = self.transitions.get("next_observations", None)
            self.actions = self.transitions.get("actions", None)
            self.rewards = self.transitions.get("rewards", None)
            self.dones = self.transitions["dones"]
            self.terminated = self.transitions.get("terminated", None)
        self.num_slots = num_slots
