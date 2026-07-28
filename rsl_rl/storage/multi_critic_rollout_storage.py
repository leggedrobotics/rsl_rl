# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.utils import split_and_pad_trajectories


class MultiCriticRolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`RolloutStorage.add_transition` to record the data.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: TensorDict | None = None
            """Observations at the current step."""

            self.actions: torch.Tensor | None = None
            """Actions taken at the current step."""

            self.rewards: torch.Tensor | None = None
            """Rewards received after the action."""

            self.dones: torch.Tensor | None = None
            """Done flags indicating episode termination."""

            # For reinforcement learning
            self.values: tuple[torch.Tensor, ...] | None = None
            """Value estimates at the current step, one per critic (RL only)."""

            self.actions_log_prob: torch.Tensor | None = None
            """Log probability of the taken actions (RL only)."""

            self.distribution_params: tuple[torch.Tensor, ...] | None = None
            """Parameters of the action distribution (RL only)."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = None
            """Privileged (teacher) actions (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState | None, ...] = ()
            """Hidden states for recurrent networks: (actor, critic_0, critic_1, ..., critic_{n-1})."""

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    class Batch:
        """A batch of data yielded by the rollout storage generators.

        This class provides named access to mini-batch fields. Fields are optional to support different training modes
        (RL vs distillation) and architectures (feedforward vs recurrent).

        For RL, ``values`` / ``advantages`` / ``returns`` are tuples with one tensor per critic (mirroring
        :class:`MultiCriticPPO`'s expectations), not single tensors.
        """

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: tuple[torch.Tensor, ...] | None = None,
            advantages: tuple[torch.Tensor, ...] | None = None,
            returns: tuple[torch.Tensor, ...] | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState | None, ...] = (),
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
        ) -> None:
            """Initialize a batch container over rollout data."""
            self.observations: TensorDict | None = observations
            """Batch of observations."""

            # For reinforcement learning
            self.actions: torch.Tensor | None = actions
            """Batch of actions."""

            self.values: tuple[torch.Tensor, ...] | None = values
            """Batch of value estimates, one tensor per critic (RL only)."""

            self.advantages: tuple[torch.Tensor, ...] | None = advantages
            """Batch of advantage estimates, one tensor per critic (RL only)."""

            self.returns: tuple[torch.Tensor, ...] | None = returns
            """Batch of return targets, one tensor per critic (RL only)."""

            self.old_actions_log_prob: torch.Tensor | None = old_actions_log_prob
            """Batch of log probabilities of the old actions (RL only)."""

            self.old_distribution_params: tuple[torch.Tensor, ...] | None = old_distribution_params
            """Batch of parameters of the old action distribution (RL only)."""

            # For distillation
            self.privileged_actions: torch.Tensor | None = privileged_actions
            """Batch of privileged (teacher) actions (distillation only)."""

            self.dones: torch.Tensor | None = dones
            """Batch of done flags (distillation only)."""

            # For recurrent networks
            self.hidden_states: tuple[HiddenState | None, ...] = hidden_states
            """Batch of hidden states for recurrent networks: (actor, critic_0, ..., critic_{n-1})."""

            self.masks: torch.Tensor | None = masks
            """Batch of trajectory masks for recurrent networks (RL recurrent only)."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        num_critics: int = 1,
        device: str = "cpu",
    ) -> None:
        """Allocate rollout buffers for a specific training mode and batch shape."""
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.num_critics = num_critics

        # Core
        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        # NOTE: values/returns/advantages are lists with one tensor per critic (each shaped
        # [num_transitions_per_env, num_envs, 1]), since MultiCriticPPO tracks a separate value
        # function / advantage / return stream per critic even though rewards are shared.
        if training_type == "rl":
            self.values: list[torch.Tensor] = [
                torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in range(num_critics)
            ]
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.distribution_params: tuple[torch.Tensor, ...] | None = None  # Lazily initialized on first transition
            self.returns: list[torch.Tensor] = [
                torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in range(num_critics)
            ]
            self.advantages: list[torch.Tensor] = [
                torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in range(num_critics)
            ]

        # For recurrent networks
        self.saved_hidden_state_a: list[torch.Tensor] | None = None
        self.saved_hidden_state_c: list[list[torch.Tensor] | None] = [None] * num_critics

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Add one transition to the storage at the current step index."""
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)  # type: ignore

        # For reinforcement learning
        if self.training_type == "rl":
            if len(transition.values) != self.num_critics:  # type: ignore
                raise ValueError(
                    f"Expected {self.num_critics} critic value estimates, got {len(transition.values)}."  # type: ignore
                )
            for c_idx, v in enumerate(transition.values):  # type: ignore
                self.values[c_idx][self.step].copy_(v.view(-1, 1))
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            if self.distribution_params is None:  # Initialize the distribution parameters
                self.distribution_params = tuple(
                    torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                    for p in transition.distribution_params  # type: ignore
                )
            for i, p in enumerate(transition.distribution_params):  # type: ignore
                self.distribution_params[i][self.step].copy_(p)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0

    # For distillation
    def generator(self) -> Generator[Batch, None, None]:
        """Yield per-timestep batches for distillation training."""
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield MultiCriticRolloutStorage.Batch(
                observations=self.observations[i],  # type: ignore
                privileged_actions=self.privileged_actions[i],
                dones=self.dones[i],
            )

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches for feedforward RL updates."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten the data. Values/returns/advantages stay as one flattened tensor per critic.
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = tuple(v.flatten(0, 1) for v in self.values)
        returns = tuple(r.flatten(0, 1) for r in self.returns)
        advantages = tuple(a.flatten(0, 1) for a in self.advantages)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Yield the mini-batch
                yield MultiCriticRolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=tuple(v[batch_idx] for v in values),
                    advantages=tuple(a[batch_idx] for a in advantages),
                    returns=tuple(r[batch_idx] for r in returns),
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield trajectory mini-batches with masks and recurrent hidden states."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        mini_batch_size = self.num_envs // num_mini_batches

        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # Handle the hidden states
                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)

                # Actor
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                if self.saved_hidden_state_a is not None:
                    hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_hidden_state_a
                    ]
                    # Remove the list wrapper for GRU (single-element case)
                    hidden_state_a_batch = (
                        hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                    )
                else:
                    hidden_state_a_batch = None

                # Critics — one entry per critic, same treatment as actor
                hidden_state_c_batches = []
                for saved_hidden_state_c in self.saved_hidden_state_c:
                    if saved_hidden_state_c is not None:
                        batch = [
                            saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                            .transpose(1, 0)
                            .contiguous()
                            for saved_hidden_state in saved_hidden_state_c
                        ]
                        batch = batch[0] if len(batch) == 1 else batch
                    else:
                        batch = None
                    hidden_state_c_batches.append(batch)

                # Yield the mini-batch. values/advantages/returns are per-critic tuples, each
                # sliced the same way the single-critic tensors used to be.
                yield MultiCriticRolloutStorage.Batch(
                    observations=padded_obs_trajectories[:, first_traj:last_traj],  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=tuple(v[:, start:stop] for v in self.values),
                    advantages=tuple(a[:, start:stop] for a in self.advantages),
                    returns=tuple(r[:, start:stop] for r in self.returns),
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    hidden_states=(hidden_state_a_batch, *hidden_state_c_batches),  # type: ignore
                    masks=trajectory_masks[:, first_traj:last_traj],
                )

                first_traj = last_traj

    def _save_hidden_states(self, hidden_states: tuple[HiddenState | None, ...]) -> None:
        """Save recurrent hidden states to the rollout storage.

        Expects ``hidden_states`` as ``(actor_hidden_state, critic_0_hidden_state, ..., critic_{n-1}_hidden_state)``.
        """
        if not hidden_states or all(h is None for h in hidden_states):
            return

        actor_hidden_state, *critic_hidden_states = hidden_states

        # --- Actor ---
        if actor_hidden_state is not None:
            hidden_state_a = (
                actor_hidden_state if isinstance(actor_hidden_state, tuple) else (actor_hidden_state,)
            )
            saved_state_a = self.saved_hidden_state_a
            if saved_state_a is None:
                saved_state_a = [
                    torch.zeros(self.num_transitions_per_env, *h.shape, device=self.device) for h in hidden_state_a
                ]
                self.saved_hidden_state_a = saved_state_a
            for i, h in enumerate(hidden_state_a):
                saved_state_a[i][self.step].copy_(h)

        # --- Critics ---
        if self.saved_hidden_state_c is None or len(self.saved_hidden_state_c) != len(critic_hidden_states):
            self.saved_hidden_state_c = [None] * len(critic_hidden_states)

        for c_idx, critic_hidden_state in enumerate(critic_hidden_states):
            if critic_hidden_state is None:
                continue
            hidden_state_c = (
                critic_hidden_state if isinstance(critic_hidden_state, tuple) else (critic_hidden_state,)
            )
            # Initialize storage lazily on first transition for this critic
            saved_state_c = self.saved_hidden_state_c[c_idx]
            if saved_state_c is None:
                saved_state_c = [
                    torch.zeros(self.num_transitions_per_env, *h.shape, device=self.device) for h in hidden_state_c
                ]
                self.saved_hidden_state_c[c_idx] = saved_state_c
            # Copy the states
            for i, h in enumerate(hidden_state_c):
                saved_state_c[i][self.step].copy_(h)