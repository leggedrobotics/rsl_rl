# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fit physical dynamics from replay and use them for differentiable rollouts."""

from __future__ import annotations

import functools
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn

from rsl_rl.env.differentiable_task import (
    DifferentiableTask,
    RolloutState,
    State,
    StateRepresentation,
    concatenate_observation_set,
)
from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_class
from rsl_rl.utils.utils import normalize_loss_across_ranks


class LearnedDynamics:
    """Learn physical transitions from simulator experience and generate differentiable trajectories.

    Reference:
        SuperMimic. arXiv:2610.28480 (2026). https://arxiv.org/abs/2610.28480
    """

    def __init__(
        self,
        task: DifferentiableTask,
        obs_groups: dict[str, list[str]],
        state: State,
        context: torch.Tensor,
        action_history: torch.Tensor,
        history_length: int,
        model: dict,
        num_warmup_steps: int = 64,
        num_warmup_iterations: int = 500,
        num_updates: int = 10,
        window_length: int = 8,
        batch_size: int = 2048,
        mirror_fraction: float = 0.0,
        loss_reduction: str = "sum",
        full_windows: bool = False,
        replay_buffer_capacity: int = 1_000_000,
        max_replay_memory_fraction: float | None = None,
        use_mixed_precision: bool = False,
        world_size: int = 1,
        device: str = "cpu",
    ) -> None:
        """Build the predictor from a sample state and allocate replay on the first real step."""
        if full_windows and num_warmup_steps <= window_length:
            raise ValueError("Complete replay windows need window_length + 1 stored states before learning.")
        if loss_reduction not in ("sum", "mean"):
            raise ValueError("loss_reduction must be 'sum' or 'mean'.")
        self._mirror_fields = {
            "mirror_state": ("state", "next_state"),
            "mirror_actions": ("actions", "action_history"),
            "mirror_context": ("context",),
        }
        if task.num_labels:
            self._mirror_fields["mirror_labels"] = ("labels",)
        if mirror_fraction:
            missing_hooks = [
                name for name in self._mirror_fields if getattr(type(task), name) is getattr(DifferentiableTask, name)
            ]
            if missing_hooks:
                raise ValueError(f"Mirror augmentation requires a task that implements {', '.join(missing_hooks)}.")

        self.task = task
        self.obs_groups = obs_groups
        self.history_length = history_length
        self.num_warmup_steps = num_warmup_steps
        self.num_warmup_iterations = num_warmup_iterations
        self.num_updates = num_updates
        self.window_length = window_length
        self.batch_size = batch_size
        self.num_mirrored_windows = round(mirror_fraction * batch_size)
        self.loss_reduction = loss_reduction
        self.full_windows = full_windows
        self.world_size = world_size
        self._autocast = functools.partial(
            torch.autocast, torch.device(device).type, dtype=torch.bfloat16, enabled=use_mixed_precision
        )

        self.error_floor = task.representation.error_floor.to(device)
        if self.error_floor.shape != (task.representation.increment_dim,):
            raise ValueError("State error_floor must contain one value per increment coordinate.")
        model_class, model_cfg = resolve_class(model)
        self.world_model = model_class(
            task.representation,
            task.representation.features(state).shape[-1],
            task.dynamics_actions(action_history[:, -1]).shape[-1],
            task.dynamics_context(state, context, action_history).shape[-1],
            num_labels=task.num_labels,
            **model_cfg,
        ).to(device)
        self.model_forward = self.world_model
        self.storage = ReplayBuffer(
            action_history.shape[0],
            replay_buffer_capacity,
            min_steps_per_env=max(history_length, num_warmup_steps),
            max_memory_fraction=max_replay_memory_fraction,
            device=device,
        )
        self.nonfinite_final_obs = torch.zeros(2, dtype=torch.bool, device=device)
        self._window_break_sum = torch.zeros((), device=device)
        self.num_collected_steps = 0

    @property
    def ready(self) -> bool:
        """Return whether replay holds enough real steps for a learner update."""
        return self.storage.num_steps >= self.num_warmup_steps

    def observe(self, transition: TensorDict, obs: TensorDict, final: TensorDict, done: torch.Tensor) -> None:
        """Store physical successors and update model normalization from real transitions."""
        groups = self.obs_groups
        successor = concatenate_observation_set(final, groups, "state")
        window_break = (concatenate_observation_set(obs, groups, "state") != successor).any(dim=-1) & ~done
        # Tasks without event labels store an empty label set
        labels = concatenate_observation_set(final, groups, "labels")
        stored = transition.clone(recurse=False)
        stored.update({"next_state": successor, "labels": labels, "dones": done, "window_break": window_break})

        # Record nonfinite consumed final observations without synchronizing; update() rejects them before training
        self.nonfinite_final_obs.logical_or_(
            torch.stack((~torch.isfinite(successor).all(), ~torch.isfinite(labels).all()))
        )
        self.storage.write_transition(stored)

        physical = self.task.pre_dynamics(
            self.task.representation.unflatten(transition["state"]), transition["context"]
        )
        self.world_model.update_normalization(
            physical,
            self.task.dynamics_actions(transition["actions"]),
            self.task.dynamics_context(physical, transition["context"], transition["action_history"]),
            self.task.representation.unflatten(successor),
        )

        self._window_break_sum.add_(window_break.float().mean())

    def advance(self) -> None:
        """Count the collected step after its device-side replay write."""
        self.storage.advance()
        self.num_collected_steps += 1

    def collection_metrics(self) -> dict[str, torch.Tensor]:
        """Report external state edits and reject invalid model targets on every rank."""
        invalid = self.nonfinite_final_obs
        if self.world_size > 1:
            invalid = invalid.clone()
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        sets = [name for name, flag in zip(("state", "labels"), invalid.tolist()) if flag]
        if sets:
            raise ValueError(f"SHAC received nonfinite final_obs in the observation sets {sets}.")
        if not self.num_collected_steps:
            return {}
        fraction = self._window_break_sum / self.num_collected_steps
        self.num_collected_steps = 0
        self._window_break_sum.zero_()
        return {"window_break_fraction": fraction}

    def clear(self) -> None:
        """Discard replay and pending collection statistics after loading a checkpoint."""
        self.storage.clear()
        self.num_collected_steps = 0
        self.nonfinite_final_obs.zero_()
        self._window_break_sum.zero_()

    def start_rollout(self, batch_size: int) -> RolloutState:
        """Build reset-aware actor histories, repeating the earliest available frame when needed."""
        history, valid = self.storage.sample_sequence(
            batch_size, ("state", "goal", "context", "action_history"), history_length=self.history_length - 1
        )

        state = self.task.representation.unflatten(history["state"])
        observations = self.task.actor_obs(state, history["goal"], history["context"], history["action_history"])
        frames = self.task.add_noise(observations["policy"])

        first_valid_offset = valid.long().argmax(dim=1)
        first_frame = frames[torch.arange(batch_size, device=frames.device), first_valid_offset]
        frames = torch.where(valid[..., None], frames, first_frame[:, None])
        return RolloutState(
            tuple(value[:, -1] for value in state),
            history["goal"][:, -1],
            history["context"][:, -1],
            history["action_history"][:, -1],
            frames,
            observations[:, -1],
        )

    def step(
        self, rollout: RolloutState, actions: torch.Tensor, survival: torch.Tensor
    ) -> tuple[RolloutState, torch.Tensor, torch.Tensor]:
        """Predict a successor and evaluate the task reward and termination on the differentiable transition."""
        state, goal, context, action_history, obs_history, _ = rollout
        transition_start = self.task.pre_dynamics(state, context)
        successor, _, labels = self._predict_transition(transition_start, actions, context, action_history)
        reward = self.task.reward(transition_start, actions, successor, goal, action_history, labels=labels)
        terminal = self.task.termination_prob(successor, goal, labels=labels)
        reachable = survival * (1 - terminal) > 0

        state = tuple(_advance_reachable_rows(reachable, new, old) for new, old in zip(successor, state))
        next_goal, next_context = self.task.advance(state, goal, context)
        goal = _advance_reachable_rows(reachable, next_goal, goal)
        context = _advance_reachable_rows(reachable, next_context, context)
        action_history = _advance_reachable_rows(
            reachable, torch.cat((action_history[:, 1:], actions[:, None]), dim=1), action_history
        )
        observations = self.task.actor_obs(state, goal, context, action_history)
        frame = self.task.add_noise(observations["policy"])
        obs_history = _advance_reachable_rows(
            reachable, torch.cat((obs_history[:, 1:], frame[:, None]), dim=1), obs_history
        )
        return RolloutState(state, goal, context, action_history, obs_history, observations), reward, terminal

    def complete_window_counts(self) -> torch.Tensor | None:
        """Prepare the eligible starts once per collection when training on complete windows."""
        return self.storage.compute_complete_window_counts(self.window_length + 1) if self.full_windows else None

    def compute_loss(
        self, cumulative_counts: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute masked state and label losses over autoregressive predictions from replay."""
        window, mask = self._sample_world_model_windows(self.batch_size, cumulative_counts)
        state = self.task.representation.unflatten(window["state"][:, 0])
        predictions, logits = [], []

        for step in range(self.window_length):
            state = self.task.pre_dynamics(state, window["context"][:, step])
            state, event_logits, _ = self._predict_transition(
                state, window["actions"][:, step], window["context"][:, step], window["action_history"][:, step]
            )
            predictions.append(state)
            if event_logits is not None:
                logits.append(event_logits)

        prediction = tuple(torch.stack(values, dim=1) for values in zip(*predictions))
        target = self.task.representation.unflatten(window["next_state"])
        scale = torch.maximum(self.world_model.increment_normalizer.std, self.error_floor)
        errors = self.task.world_model_loss(prediction, target, scale)
        masked_errors = _mask_loss(errors, mask, "world_model_loss")
        valid_count = mask.sum()
        mean_error = normalize_loss_across_ranks(masked_errors.sum(), valid_count, self.world_size)
        loss = mean_error if self.loss_reduction == "mean" else masked_errors.sum(dim=1).mean()
        metrics = {"world_model_error": mean_error.detach()}

        if self.task.num_labels:
            label_errors = self.task.label_loss(torch.stack(logits, 1).float(), window["labels"].float())
            masked_labels = _mask_loss(label_errors, mask, "label_loss")
            label_loss = normalize_loss_across_ranks(masked_labels.sum(), valid_count, self.world_size)
            loss = loss + label_loss
            metrics["label_loss"] = label_loss.detach()

        metrics["world_model_loss"] = loss.detach() / (self.window_length if self.loss_reduction == "sum" else 1)
        return loss, metrics

    def _sample_world_model_windows(
        self, batch_size: int, cumulative_counts: torch.Tensor | None = None
    ) -> tuple[TensorDict, torch.Tensor]:
        """Gather dynamics training windows and mirror the configured share."""
        keys = ("state", "next_state", "actions", "context", "labels", "action_history")
        if self.full_windows and cumulative_counts is None:
            cumulative_counts = self.complete_window_counts()
        window, valid = self.storage.sample_sequence(
            batch_size, keys, self.window_length, complete_window_counts=cumulative_counts
        )

        count = self.num_mirrored_windows
        if count:
            for hook, fields in self._mirror_fields.items():
                mirror = getattr(self.task, hook)
                for key in fields:
                    values = window[key]
                    window[key] = torch.cat((mirror(values[:count]), values[count:]))

        return window, valid.float()

    def _predict_transition(
        self, state: State, actions: torch.Tensor, context: torch.Tensor, action_history: torch.Tensor
    ) -> tuple[State, torch.Tensor | None, torch.Tensor | None]:
        """Apply learned dynamics, successor labels and task-owned analytic state."""
        with self._autocast():
            raw_next = self.model_forward(
                state, self.task.dynamics_actions(actions), self.task.dynamics_context(state, context, action_history)
            )

            # Successor events depend on the action just applied, so their history includes it
            logits = None
            if self.task.num_labels:
                next_action_history = torch.cat((action_history[..., 1:, :], actions.unsqueeze(-2)), dim=-2)
                logits = self.world_model.predict_label_logits(
                    raw_next, self.task.dynamics_context(raw_next, context, next_action_history)
                )

        labels = self.task.label_activation(logits.float()) if logits is not None else None
        return self.task.post_dynamics(state, raw_next, labels, actions, context), logits, labels


class WorldModel(nn.Module):
    """Predict the next physical state and optional event labels."""

    def __init__(
        self,
        representation: StateRepresentation,
        feature_dim: int,
        num_actions: int,
        context_dim: int,
        hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
        num_labels: int = 0,
        label_hidden_dims: tuple[int, ...] = (512, 512),
        activation: str = "swish",
        normalize_output: bool = True,
        normalize_context: bool = True,
        zero_initialize_output: bool = True,
    ) -> None:
        """Build normalized residual dynamics and an optional event head."""
        super().__init__()
        self.representation = representation
        self.feature_dim = feature_dim
        self.normalize_output = normalize_output
        self.normalize_context = normalize_context
        self._context_start = feature_dim + num_actions

        self.input_normalizer = EmpiricalNormalization(self._context_start + (context_dim if normalize_context else 0))
        self.increment_normalizer = EmpiricalNormalization(representation.increment_dim, eps=1e-4)

        self.dynamics_mlp = MLP(
            self._context_start + context_dim,
            representation.increment_dim,
            hidden_dims,
            activation=activation,
            norm="layer",
        )
        if zero_initialize_output:
            nn.init.zeros_(self.dynamics_mlp.output_layer.weight)
            nn.init.zeros_(self.dynamics_mlp.output_layer.bias)

        self.label_head = (
            MLP(feature_dim + context_dim, num_labels, label_hidden_dims, norm="layer") if num_labels else None
        )

    def forward(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> State:
        """Integrate the predicted increment while preserving state and action derivatives."""
        inputs = self._concatenate_dynamics_inputs(state, actions, context)
        if self.normalize_context:
            inputs = self.input_normalizer(inputs)
        else:
            inputs = torch.cat((self.input_normalizer(inputs[..., : self._context_start]), context), dim=-1)

        increment = self.dynamics_mlp(inputs).float()
        if self.normalize_output:
            increment = self.increment_normalizer.inverse(increment)
        with torch.autocast(increment.device.type, enabled=False):
            return self.representation.integrate(state, increment)

    def update_normalization(
        self, state: State, actions: torch.Tensor, context: torch.Tensor, successor: State
    ) -> None:
        """Update real-transition input and increment statistics at collection time."""
        inputs = self._concatenate_dynamics_inputs(state, actions, context)
        self.input_normalizer.update(inputs if self.normalize_context else inputs[..., : self._context_start])
        self.increment_normalizer.update(self.representation.increments(state, successor))

    def predict_label_logits(self, state: State, context: torch.Tensor) -> torch.Tensor:
        """Predict successor events using the matching physical/context input moments."""
        if self.label_head is None:
            raise ValueError("The world model has no event label head.")

        mean = self.input_normalizer.mean
        std = self.input_normalizer.std
        eps = self.input_normalizer.eps
        features = self.representation.features(state)
        features = (features - mean[: self.feature_dim]) / (std[: self.feature_dim] + eps)
        if self.normalize_context:
            context = (context - mean[self._context_start :]) / (std[self._context_start :] + eps)
        return self.label_head(torch.cat((features, context), dim=-1))

    def _concatenate_dynamics_inputs(self, state: State, actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Concatenate the physical features, actions and context for dynamics prediction."""
        return torch.cat((self.representation.features(state), actions, context), dim=-1)


def _mask_loss(errors: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a validity mask to task losses without broadcasting across replay positions."""
    if errors.shape != mask.shape:
        raise ValueError(
            f"{name} must return one value per replay position, expected {mask.shape}, got {errors.shape}."
        )
    return errors * mask


def _advance_reachable_rows(reachable: torch.Tensor, successor: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Advance only the rows that can reach the successor."""
    return torch.where(reachable.reshape(-1, *([1] * (successor.ndim - 1))), successor, current)
