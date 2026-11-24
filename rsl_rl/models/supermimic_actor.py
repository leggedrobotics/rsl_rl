# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic's deterministic policy for task-provided frames and goals."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.modules import MLP, EmpiricalNormalization


class _SuperMimicPolicy(nn.Module):
    """Normalize all frames with shared moments and decode an action conditioned on the current goal."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        history_length: int,
        num_actions: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...],
        decoder_hidden_dims: tuple[int, ...],
        activation: str,
        norm: str,
        encoder_norm: str,
        output_gain: float,
    ) -> None:
        """Construct the goal encoder, history decoder and input normalizers."""
        super().__init__()
        self.obs_dim, self.goal_dim = obs_dim, goal_dim
        self.history_length, self.num_actions = history_length, num_actions
        self.encoder = (
            MLP(goal_dim, latent_dim, encoder_hidden_dims, activation=activation, norm=encoder_norm)
            if goal_dim
            else nn.Identity()
        )
        self.decoder = MLP(
            obs_dim * history_length + (latent_dim if goal_dim else 0),
            num_actions,
            decoder_hidden_dims,
            activation=activation,
            norm=norm,
        )
        with torch.no_grad():
            self.decoder.output_layer.weight.mul_(output_gain)
            if output_gain != 1.0:
                self.decoder.output_layer.bias.zero_()
        self.obs_norm = EmpiricalNormalization(obs_dim)
        self.goal_norm = EmpiricalNormalization(goal_dim) if goal_dim else nn.Identity()

    def forward(self, obs_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return actions from an observation history ``[..., history_length, obs_dim]`` and a goal, if any."""
        features = self.obs_norm(obs_history).flatten(-2)
        if self.goal_dim:
            encoded = torch.tanh(self.encoder(self.goal_norm(goal)))
            features = torch.cat((encoded, features), dim=-1)
        return self.decoder(features)


class SuperMimicActor(nn.Module):
    """Decode actions from normalized frame history and an optional encoded goal.

    The actor observation set contains the current ``policy`` frame and optionally the flattened preceding
    ``history`` in chronological order. Additional groups form the deployable goal in their configured order.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        latent_dim: int = 32,
        encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
        activation: str = "elu",
        norm: str = "none",
        encoder_norm: str = "layer",
        output_gain: float = 0.01,
    ) -> None:
        """Infer the frame, history and goal widths from the environment's actor observations."""
        super().__init__()
        self.obs_groups = obs_groups[obs_set]
        if "policy" not in self.obs_groups:
            raise ValueError("SuperMimicActor requires a current 'policy' frame in its actor observation set.")
        self.goal_groups = [key for key in self.obs_groups if key not in ("policy", "history")]
        frame_dim = obs["policy"].shape[-1]
        history_length = 1
        if "history" in self.obs_groups:
            history_length += obs["history"].unflatten(-1, (-1, frame_dim)).shape[-2]
        self.policy = _SuperMimicPolicy(
            frame_dim,
            sum(obs[key].shape[-1] for key in self.goal_groups),
            history_length,
            output_dim,
            latent_dim,
            encoder_hidden_dims,
            decoder_hidden_dims,
            activation,
            norm,
            encoder_norm,
            output_gain,
        )
        self.export_reference_action = False

    def _goal(self, obs: TensorDict) -> torch.Tensor:
        """Join the deployable goal groups for this frame."""
        if self.goal_groups:
            return torch.cat([obs[key] for key in self.goal_groups], dim=-1)
        return obs["policy"][..., :0]

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Apply the same differentiable policy during collection and imagination."""
        frame = obs["policy"]
        if self.policy.history_length > 1:
            history = torch.cat((obs["history"], frame), dim=-1).unflatten(
                -1, (self.policy.history_length, self.policy.obs_dim)
            )
        else:
            history = frame.unsqueeze(-2)
        return self.policy(history, self._goal(obs)).float()

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        """Update shared frame and goal statistics once per real observation."""
        self.policy.obs_norm.update(obs["policy"])
        if self.policy.goal_dim:
            self.policy.goal_norm.update(self._goal(obs))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Frame history is supplied by the environment at each step."""

    def as_jit(self) -> nn.Module:
        """Export the original explicit history, optional goal and optional reference inputs."""
        if self.export_reference_action:
            return _GoalReferenceExport(self.policy) if self.policy.goal_dim else _ReferenceExport(self.policy)
        return _GoalExport(self.policy) if self.policy.goal_dim else _HistoryExport(self.policy)

    def as_onnx(self) -> nn.Module:
        """Export the same policy and input order as TorchScript."""
        return self.as_jit()


class _HistoryExport(nn.Module):
    """Stateless export of a copied policy network taking only the observation history.

    TorchScript needs a fixed signature per input combination, so each subclass defines ``forward`` for its inputs.
    """

    def __init__(self, policy: _SuperMimicPolicy) -> None:
        super().__init__()
        self.policy = copy.deepcopy(policy)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Return an action for the supplied frame history."""
        return self.policy(obs_history, obs_history[..., 0, :0])

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative inputs in the order of the deployment input names."""
        shapes = {
            "obs_history": (self.policy.history_length, self.policy.obs_dim),
            "goal": (self.policy.goal_dim,),
            "reference_action": (self.policy.num_actions,),
        }
        return tuple(torch.zeros(1, *shapes[name]) for name in self.input_names)

    @property
    def input_names(self) -> list[str]:
        """Return deployment input names."""
        return ["obs_history"]

    @property
    def output_names(self) -> list[str]:
        """Return the action output name."""
        return ["actions"]


class _GoalExport(_HistoryExport):
    """Stateless export with a separate deployable goal input."""

    def forward(self, obs_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return an action from explicit history and encoded goal."""
        return self.policy(obs_history, goal)

    @property
    def input_names(self) -> list[str]:
        """Return deployment input names."""
        return ["obs_history", "goal"]


class _ReferenceExport(_HistoryExport):
    """Stateless export that adds the policy output to a reference action."""

    def forward(self, obs_history: torch.Tensor, reference_action: torch.Tensor) -> torch.Tensor:
        """Return the reference action corrected by the policy."""
        return self.policy(obs_history, obs_history[..., 0, :0]) + reference_action

    @property
    def input_names(self) -> list[str]:
        """Return deployment input names."""
        return ["obs_history", "reference_action"]


class _GoalReferenceExport(_HistoryExport):
    """Stateless export with a goal input that adds the policy output to a reference action."""

    def forward(self, obs_history: torch.Tensor, goal: torch.Tensor, reference_action: torch.Tensor) -> torch.Tensor:
        """Return the reference action corrected by the goal-conditioned policy."""
        return self.policy(obs_history, goal) + reference_action

    @property
    def input_names(self) -> list[str]:
        """Return deployment input names."""
        return ["obs_history", "goal", "reference_action"]
