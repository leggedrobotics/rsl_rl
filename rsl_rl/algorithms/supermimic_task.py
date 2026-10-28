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
        """Build one noise-free deployable observation frame."""
        raise NotImplementedError

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Encode a deployable policy goal, preserving state derivatives."""
        raise NotImplementedError

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance exogenous goals and privileged context after an imagined physical transition."""
        raise NotImplementedError

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Return normalized privileged dynamics inputs."""
        raise NotImplementedError

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Build privileged value features for the synthetic critic observation set."""
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

    def post_dynamics(self, state: State, next_state: State, labels: torch.Tensor | None) -> State:
        """Advance task-owned analytic state after integration and successor-label prediction."""
        return next_state

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        """Add task observation noise while retaining imagined state derivatives."""
        return observation
