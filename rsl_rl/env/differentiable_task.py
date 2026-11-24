# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned differentiable state coordinates and heads for SHAC."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from typing import NamedTuple, Protocol

State = tuple[torch.Tensor, ...]


class RolloutState(NamedTuple):
    """Physical state and feature histories at one step of a differentiable rollout."""

    state: State
    goal: torch.Tensor
    context: torch.Tensor
    action_history: torch.Tensor
    obs_history: torch.Tensor
    observations: TensorDict

    @property
    def actor_obs(self) -> TensorDict:
        """Return policy inputs for this rollout step, including its noisy observation history."""
        return prepare_actor_obs(self.observations, self.obs_history)


def prepare_actor_obs(observations: TensorDict, history: torch.Tensor) -> TensorDict:
    """Replace the policy frame with the current noisy frame and expose any preceding frames oldest first."""
    inputs = observations.clone(recurse=False)
    inputs["policy"] = history[..., -1, :]
    if history.shape[-2] > 1:
        inputs["history"] = history[..., :-1, :].flatten(-2)
    return inputs


def concatenate_observation_set(obs: TensorDict, obs_groups: dict[str, list[str]], name: str) -> torch.Tensor:
    """Concatenate one explicitly mapped observation set, using an empty tensor for absent optional sets."""
    groups = obs_groups.get(name)
    if groups:
        return torch.cat([obs[key] for key in groups], dim=-1)
    return obs[obs_groups["state"][0]].new_empty((*obs.batch_size, 0))


class StateRepresentation(Protocol):
    """Convert between task observations and differentiable physical coordinates."""

    increment_dim: int
    """Number of coordinates predicted by the world model."""

    error_floor: torch.Tensor
    """Positive minimum loss scale for each increment coordinate."""

    def unflatten(self, flat: torch.Tensor) -> State:
        """Decode the last dimension of a physical observation, preserving leading batch dimensions."""
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
    """Provide task features, rewards and state updates for SHAC.

    ``prev_actions`` has shape ``[..., action_history_length, num_actions]``, oldest first. Actor observations and
    goals must use deployable inputs. Privileged context is available separately to dynamics and critic heads.
    Heads and the representation need finite gradients as well as finite values, e.g. ``torch.linalg.vector_norm``
    instead of ``sqrt(sum(x**2))`` at zero: an optimizer step with a nonfinite gradient is skipped and shows as a
    nonfinite gradient norm in the logs.
    """

    representation: StateRepresentation
    action_history_length: int = 2
    num_labels: int = 0

    def start_rollout(self, batch_size: int) -> RolloutState:
        """Provide initial states and actor histories when using task dynamics directly."""
        raise NotImplementedError

    def step(
        self, rollout: RolloutState, actions: torch.Tensor, survival: torch.Tensor
    ) -> tuple[RolloutState, torch.Tensor, torch.Tensor]:
        """Advance task dynamics and return the next rollout state, reward and termination probability.

        Preserve action derivatives and leave unreachable rows at their last reachable state.
        """
        raise NotImplementedError

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> TensorDict:
        """Return named deployable observations for this state, preserving its leading batch dimensions and gradients.

        Match the groups the environment exposes to the actor. Learned dynamics applies noise to the ``policy``
        frame and constructs its ``history``. A command can be part of the frame or a separate group.
        """
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
        """Map policy outputs to physical commands before world-model normalization, e.g. scaled joint targets."""
        return actions

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Return normalized privileged dynamics inputs, e.g. log-scaled randomized masses and the recent actions."""
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
        """Return transition rewards from the :meth:`pre_dynamics` start, pre-transition goal and successor labels."""
        raise NotImplementedError

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return successor termination probabilities in [0,1].

        For example from predicted contact labels at illegal bodies, or from a soft threshold on a tracking error.
        """
        raise NotImplementedError

    def pre_dynamics(self, state: State, context: torch.Tensor) -> State:
        """Apply an exogenous edit to a transition start after the actor has acted, e.g. a scheduled velocity impulse.

        The edit must be a deterministic function of ``state`` and ``context`` without random draws, because it runs
        wherever the world-model inputs of a transition are built: for the world-model statistics during collection,
        in each step of the world-model fit and in each imagined step. Replay keeps the state the actor observed, and
        the actor and :meth:`critic_obs` see that state. :meth:`dynamics_context`, the world model,
        :meth:`post_dynamics` and :meth:`reward` receive the edited start. Imagined returns are differentiated through
        the edit, so it must preserve state derivatives.
        """
        return state

    def post_dynamics(
        self, state: State, next_state: State, labels: torch.Tensor | None, actions: torch.Tensor, context: torch.Tensor
    ) -> State:
        """Advance task-owned analytic state after integration and successor-label prediction.

        ``state`` is the transition start returned by :meth:`pre_dynamics`. ``actions`` are the policy actions applied
        in the transition, before :meth:`dynamics_actions`, and ``context`` is its privileged context, for analytic
        state that they drive, such as filters, delay buffers or estimators.
        """
        return next_state

    def label_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Map successor-label logits to the labels read by rewards, terminations and post-dynamics.

        The default yields event probabilities. Override it together with :meth:`label_loss`, e.g. to regress labels.
        """
        return torch.sigmoid(logits)

    def label_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return one successor-label loss per replay position; the default treats labels as binary events.

        Reduce the label dimension only: the learner masks invalid replay positions and rejects any other shape.
        """
        return nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=-1)

    def reference_action(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Return the action that the policy output corrects when the actor uses ``use_reference_action``.

        The environment adds this reference to the correction returned by the actor. SHAC uses the same
        reference to record applied actions in replay and to step the imagined dynamics.
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
        """Return the mirror image of successor labels; required only for tasks with labels."""
        raise NotImplementedError

    def world_model_loss(self, prediction: State, target: State, scale: torch.Tensor) -> torch.Tensor:
        """Return one dynamics loss per replay position, including any integrated state coordinates.

        Reduce the state dimensions only: the learner masks invalid replay positions and rejects any other shape.
        """
        return (self.representation.error(prediction, target) / scale).square().mean(dim=-1)

    def add_noise(self, observation: torch.Tensor) -> torch.Tensor:
        """Add task observation noise while retaining imagined state derivatives.

        This is the only source of actor observation noise: it is applied in training-mode collection and imagination,
        not in eval mode; the observation sets from the environment are expected to be noise-free.
        """
        return observation

    def update_curriculum(self, goal: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Update task-owned curricula from a completed physical step and its pre-step goal.

        For example the sampling weights of reference motions.
        """
        pass
