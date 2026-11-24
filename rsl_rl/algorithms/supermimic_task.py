# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned differentiable state coordinates and heads for SuperMimic."""

from __future__ import annotations

import torch
from typing import Protocol

State = tuple[torch.Tensor, ...]


class StateRepresentation(Protocol):
    """Differentiable state coordinates supporting arbitrary leading batch dimensions."""

    increment_dim: int
    error_floor: torch.Tensor

    def unflatten(self, flat: torch.Tensor) -> State:
        """Decode a physical observation without detaching derivatives."""
        ...

    def features(self, state: State) -> torch.Tensor:
        """Encode physical dynamics inputs."""
        ...

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Encode the per-step successor-minus-current increment in integration coordinates."""
        ...

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Apply a per-step physical increment; the algorithm performs no time-step scaling."""
        ...

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Return prediction errors in increment coordinates."""
        ...


class EuclideanStateRepresentation:
    """Flat Euclidean coordinates for systems without manifold state components."""

    def __init__(self, dimension: int, error_floor: float = 1e-3) -> None:
        """Set state width and the minimum per-coordinate model-loss scale."""
        if dimension < 1 or error_floor <= 0:
            raise ValueError("Euclidean state dimension and error floor must be positive.")
        self.increment_dim = dimension
        self.error_floor = torch.full((dimension,), error_floor)

    def unflatten(self, flat: torch.Tensor) -> State:
        """Keep the flat vector as one structured component."""
        return (flat,)

    def features(self, state: State) -> torch.Tensor:
        """Return the physical vector."""
        return state[0]

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Subtract successive physical vectors."""
        return next_state[0] - state[0]

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Add the predicted physical increment."""
        return (state[0] + increment,)

    def error(self, prediction: State, target: State) -> torch.Tensor:
        """Subtract the target physical vector."""
        return prediction[0] - target[0]


class DifferentiableTask:
    """Differentiable task heads; simulator geometry, rewards and curricula stay task-owned.

    ``prev_actions`` has shape ``[..., action_history_length, num_actions]``, oldest first. Actor observations and
    goals must use deployable inputs. Privileged context is available separately to dynamics and critic heads.
    Heads and the representation need finite gradients as well as finite values, e.g. ``torch.linalg.vector_norm``
    instead of ``sqrt(sum(x**2))`` at zero: an optimizer step with a nonfinite gradient is skipped and shows as a
    nonfinite gradient norm in the logs.
    """

    representation: StateRepresentation
    action_history_length: int = 2
    num_labels: int = 0

    def to(self, device: torch.device | str) -> DifferentiableTask:
        """Move tensor attributes used by actor heads; override for nested task state."""
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Build noise-free deployable frames; replay also passes ``[batch, history_length]`` leading dimensions."""
        raise NotImplementedError

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Encode a deployable policy goal, preserving state derivatives."""
        raise NotImplementedError

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance exogenous goals and privileged context after an imagined physical transition.

        Imagined rollouts ignore the environment's time limit: they run ``horizon`` steps from any stored step, so goals
        must stay valid that far past the last one the environment shows, e.g. by clamping a reference-frame index.
        """
        raise NotImplementedError

    def dynamics_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Map policy outputs to physical commands before world-model normalization."""
        return actions

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Return normalized privileged dynamics inputs."""
        raise NotImplementedError

    def critic_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Build privileged value features for the synthetic critic observation set.

        ``history`` is the observation history the actor acts on at ``state``, including :meth:`add_noise`, with shape
        ``[..., history_length, obs_dim]``, oldest first. During imagination it carries gradients: detach it if the
        critic must not shape the actor through it.
        """
        raise NotImplementedError

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return transition rewards using the pre-transition goal and successor event probabilities."""
        raise NotImplementedError

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return successor termination probabilities in [0,1]."""
        raise NotImplementedError

    def post_dynamics(
        self, state: State, next_state: State, labels: torch.Tensor | None, actions: torch.Tensor, context: torch.Tensor
    ) -> State:
        """Advance task-owned analytic state after integration and successor-label prediction.

        ``actions`` are the policy actions applied in the transition, before :meth:`dynamics_actions`, and ``context``
        is its privileged context, for analytic state that they drive, such as filters, delay buffers or estimators.
        """
        return next_state

    def reference_action(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Return the action that the policy output corrects when the actor uses ``use_reference_action``.

        Motion-tracking tasks can return the reference joint targets in action units, so the policy learns a
        correction to the reference pose. Use deployable inputs only: exported policies take this action as an
        additional ``reference_action`` input.
        """
        raise NotImplementedError

    def mirror_state(self, state: torch.Tensor) -> torch.Tensor:
        """Return the left-right mirror image of flat physical states for world-model mirror augmentation.

        The mirror hooks map each replayed quantity with arbitrary leading dimensions. Together they must yield
        physically valid transitions for the robot, e.g. by swapping left and right joints and negating lateral motion.
        """
        raise NotImplementedError

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the mirror image of policy actions, also applied to each previous action."""
        raise NotImplementedError

    def mirror_context(self, context: torch.Tensor) -> torch.Tensor:
        """Return the mirror image of flat privileged context, e.g. by swapping left and right body masses."""
        raise NotImplementedError

    def mirror_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Return the mirror image of successor event labels; required only for tasks with labels."""
        raise NotImplementedError

    def world_model_loss(self, prediction: State, target: State, scale: torch.Tensor) -> torch.Tensor:
        """Return one dynamics loss per replay position, including any integrated state coordinates."""
        return (self.representation.error(prediction, target) / scale).square().mean(dim=-1)

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        """Add task observation noise while retaining imagined state derivatives.

        This is the only source of actor observation noise: it is applied in training-mode collection and imagination,
        not in eval mode; the observation sets from the environment are expected to be noise-free.
        """
        return observation

    def update_curriculum(self, goal: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Update task-owned curricula from a completed physical step and its pre-step goal."""
        pass
