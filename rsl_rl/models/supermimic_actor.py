# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic actor with real-observation history and a stateless deployable network."""

from __future__ import annotations

import copy
import torch
from collections.abc import Callable
from tensordict import TensorDict
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rsl_rl.algorithms.supermimic_task import DifferentiableTask
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class SuperMimicActor(nn.Module):
    """Goal encoder and observation-history decoder, with task-owned real observation conversion.

    Real inference takes the environment TensorDict. JIT and ONNX exports take only deployable ``obs_history`` and
    optional encoded ``goal`` inputs. Deployment assembles the history and computes the same task features.
    """

    is_recurrent = False

    def __init__(
        self,
        task: DifferentiableTask,
        obs_groups: dict[str, list[str]],
        obs_dim: int,
        goal_dim: int,
        num_envs: int,
        num_actions: int,
        history_length: int = 5,
        latent_dim: int = 32,
        encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
    ) -> None:
        """Construct the networks and nonpersistent collection history."""
        super().__init__()
        if min(history_length, num_envs, num_actions, obs_dim, task.action_history_length) < 1 or goal_dim < 0:
            raise ValueError("Actor/history dimensions must be positive and goal_dim nonnegative.")
        self.task, self.obs_groups = task, obs_groups
        self.obs_dim, self.goal_dim = obs_dim, goal_dim
        self.history_length, self.num_actions = history_length, num_actions
        self.encoder = MLP(goal_dim, latent_dim, encoder_hidden_dims) if goal_dim else nn.Identity()
        self.decoder = MLP(obs_dim * history_length + (latent_dim if goal_dim else 0), num_actions, decoder_hidden_dims)
        with torch.no_grad():
            self.decoder.output.weight.mul_(0.01)
            self.decoder.output.bias.zero_()
        self.obs_norm = EmpiricalNormalization(obs_dim)
        self.goal_norm = EmpiricalNormalization(goal_dim) if goal_dim else nn.Identity()
        self.register_buffer("history", torch.zeros(num_envs, history_length, obs_dim), persistent=False)
        self.register_buffer(
            "previous_actions", torch.zeros(num_envs, task.action_history_length, num_actions), persistent=False
        )
        self.register_buffer("fresh", torch.ones(num_envs, dtype=torch.bool), persistent=False)
        self.register_buffer("noise_std", torch.zeros(num_actions), persistent=False)

    def _apply(self, fn: Callable, recurse: bool = True) -> SuperMimicActor:
        """Keep task observation tensors on the actor device."""
        super()._apply(fn, recurse=recurse)
        self.task.to(self.history.device)
        return self

    def observation_set(self, obs: TensorDict, name: str) -> torch.Tensor:
        """Concatenate one explicitly mapped set, using an empty tensor for absent optional sets."""
        groups = self.obs_groups.get(name)
        if groups:
            return torch.cat([obs[key] for key in groups], dim=-1)
        return self.history.new_empty((*obs.batch_size, 0))

    @torch.no_grad()
    def sync_actions(self, obs: TensorDict) -> None:
        """Read task-reported action history, including nonzero reset hold actions."""
        if "prev_action" in self.obs_groups:
            self.previous_actions[:, -1].copy_(self.observation_set(obs, "prev_action"))
        if "prev_prev_action" in self.obs_groups and self.previous_actions.shape[1] > 1:
            self.previous_actions[:, -2].copy_(self.observation_set(obs, "prev_prev_action"))

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Build a real actor frame, update history and return a deterministic action."""
        self.sync_actions(obs)
        state = self.task.representation.unflatten(self.observation_set(obs, "state"))
        goal, context = self.observation_set(obs, "goal"), self.observation_set(obs, "context")
        frame = self.task.actor_obs(state, goal, context, self.previous_actions)
        policy_goal = self.task.actor_goal(state, goal)
        if self.training:
            frame = self.task.add_noise(frame)
            self.obs_norm.update(frame)
            if self.goal_dim:
                self.goal_norm.update(policy_goal)
        shifted = torch.cat((self.history[:, 1:], frame.unsqueeze(1)), dim=1)
        self.history.copy_(torch.where(self.fresh[:, None, None], frame[:, None].expand_as(shifted), shifted))
        self.fresh.zero_()
        action = self.forward_history(self.history, policy_goal)
        self.previous_actions.copy_(torch.cat((self.previous_actions[:, 1:], action[:, None]), dim=1))
        return action

    def forward_history(self, obs_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Evaluate the stateless differentiable policy used during imagination."""
        history = self.obs_norm(obs_history).flatten(-2)
        if self.goal_dim:
            encoded = torch.tanh(self.encoder(self.goal_norm(goal)))
            history = torch.cat((encoded, history), dim=-1)
        return self.decoder(history).float()

    @torch.no_grad()
    def reset(
        self,
        dones: torch.Tensor | None = None,
        reset_action: torch.Tensor | None = None,
        previous_reset_action: torch.Tensor | None = None,
    ) -> None:
        """Clear observation history and seed actions from an optional simulator hold command."""
        rows = torch.ones_like(self.fresh) if dones is None else dones.reshape(-1).bool()
        self.fresh[rows] = True
        self.history[rows] = 0
        self.previous_actions[rows] = 0 if reset_action is None else reset_action[rows, None]
        if previous_reset_action is not None and self.previous_actions.shape[1] > 1:
            self.previous_actions[rows, -2] = previous_reset_action[rows]

    @property
    def output_std(self) -> torch.Tensor:
        """Return the fixed Gaussian collection-noise standard deviation."""
        return self.noise_std

    def as_jit(self) -> nn.Module:
        """Return the stateless network with explicit deployable history and optional goal inputs."""
        return _GoalActorExport(self) if self.goal_dim else _ActorExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return the same stateless representation for ONNX export."""
        return self.as_jit()


class _ActorExport(nn.Module):
    """Stateless export for an actor without a separate goal input."""

    def __init__(self, actor: SuperMimicActor) -> None:
        super().__init__()
        self.obs_norm = copy.deepcopy(actor.obs_norm)
        self.decoder = copy.deepcopy(actor.decoder)
        self.history_length, self.obs_dim = actor.history_length, actor.obs_dim

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Return an action from an explicit deployable observation history."""
        return self.decoder(self.obs_norm(obs_history).flatten(-2))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative history input."""
        return (torch.zeros(1, self.history_length, self.obs_dim),)

    @property
    def input_names(self) -> list[str]:
        """Return deployment input names."""
        return ["obs_history"]

    @property
    def output_names(self) -> list[str]:
        """Return the action output name."""
        return ["actions"]


class _GoalActorExport(_ActorExport):
    """Stateless export for an actor with a separate deployable goal input."""

    def __init__(self, actor: SuperMimicActor) -> None:
        super().__init__(actor)
        self.goal_norm = copy.deepcopy(actor.goal_norm)
        self.encoder = copy.deepcopy(actor.encoder)
        self.goal_dim = actor.goal_dim

    def forward(self, obs_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return an action from explicit history and encoded goal."""
        history = self.obs_norm(obs_history).flatten(-2)
        encoded = torch.tanh(self.encoder(self.goal_norm(goal)))
        return self.decoder(torch.cat((encoded, history), dim=-1))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return representative history and goal inputs."""
        return torch.zeros(1, self.history_length, self.obs_dim), torch.zeros(1, self.goal_dim)

    @property
    def input_names(self) -> list[str]:
        """Return deployment input names."""
        return ["obs_history", "goal"]
