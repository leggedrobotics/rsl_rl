# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code ported from FlashSAC (flash_rl):
#   Copyright (c) 2026 Holiday Robotics
#   SPDX-License-Identifier: MIT
# The actor/critic network composition is adapted from
# ``flash_rl/agents/flashSAC/network.py``. The RSL-RL model surface
# (observation-group handling, normalization, JIT/ONNX export) mirrors
# ``rsl_rl/models/mlp_model.py``.

"""FlashSAC actor and critic models.

These follow the RSL-RL model convention ``Model(obs, obs_groups, obs_set,
output_dim, **cfg)`` used by :class:`~rsl_rl.models.MLPModel`, so they slot into
the same ``construct_algorithm`` / observation-group machinery. The actor also
provides :meth:`as_jit` / :meth:`as_onnx` so the standard runner export path and
mjlab's ONNX metadata attachment work unchanged.

Unlike ``MLPModel``, the FlashSAC networks take an explicit ``training`` flag
(threaded to their internal BatchNorm layers) that is *independent* of
``nn.Module.training``. Deterministic inference and export always run with
``training=False`` so BatchNorm uses its running statistics.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.modules.flashsac_layers import FlashSACActor, FlashSACDoubleCritic


def _get_obs_dim(obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
    """Select active observation groups and compute the concatenated 1D dimension.

    Mirrors :meth:`rsl_rl.models.MLPModel._get_obs_dim`: only 1D observation
    groups are supported and their dimensions are summed.
    """
    if obs_set not in obs_groups:
        raise KeyError(
            f"Observation set '{obs_set}' not found in obs_groups {list(obs_groups.keys())}. "
            "The SAC models require the set to be resolved (see resolve_sac_obs_groups)."
        )
    active_obs_groups = obs_groups[obs_set]
    obs_dim = 0
    for obs_group in active_obs_groups:
        if len(obs[obs_group].shape) != 2:
            raise ValueError(
                f"The FlashSAC models only support 1D observations, got shape {obs[obs_group].shape} "
                f"for group '{obs_group}'."
            )
        obs_dim += obs[obs_group].shape[-1]
    return active_obs_groups, obs_dim


class FlashSACActorModel(nn.Module):
    """Squashed-Gaussian actor with RSL-RL model surface and export wrappers."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_blocks: int = 2,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        obs_normalization: bool = False,
    ) -> None:
        """Initialize the actor model, resolving observation groups and normalization."""
        super().__init__()

        self.obs_groups, self.obs_dim = _get_obs_dim(obs, obs_groups, obs_set)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.net = FlashSACActor(
            num_blocks=num_blocks,
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            action_dim=output_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

        # Cached diagnostics from the last stochastic evaluation (for logging).
        self._last_std: torch.Tensor | None = None
        self._last_log_prob: torch.Tensor | None = None

    # -- observation handling ------------------------------------------------

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate the actor observation groups and normalize."""
        latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the observation normalizer statistics (no-op if disabled)."""
        if self.obs_normalization:
            latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(latent)  # type: ignore[operator]

    # -- forward / sampling --------------------------------------------------

    def mean_std(self, obs: TensorDict, training: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pre-squash Gaussian ``(mean, std)`` for the observations."""
        latent = self.get_latent(obs)
        return self.net.get_mean_and_std(latent, training)

    def sample(self, obs: TensorDict, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a squashed action and return ``(actions, log_prob)``.

        Used by the SAC algorithm both for the actor update (``training=True``,
        so BatchNorm running stats update) and for target computation
        (``training=False``).
        """
        latent = self.get_latent(obs)
        actions, info = self.net(latent, training)
        self._last_log_prob = info["log_prob"]
        return actions, info["log_prob"]

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Deterministic (default) or stochastic inference forward.

        Deterministic by default (parity with ``MLPModel``): returns
        ``tanh(mean)``. Runs the network with ``training=False`` so BatchNorm
        uses running statistics. This is the path the viewer/play loop invokes
        via ``policy(obs)``.
        """
        del masks, hidden_state  # non-recurrent
        mean, std = self.mean_std(obs, training=False)
        self._last_std = std
        if stochastic_output:
            actions = torch.tanh(mean + std * torch.randn_like(std))
            return actions
        return torch.tanh(mean)

    # -- weight normalization ------------------------------------------------

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Renormalize all weight-normalized submodules (call after each optimizer step)."""
        for module in self.net.modules():
            if hasattr(module, "normalize_parameters"):
                module.normalize_parameters()  # type: ignore[operator]

    # -- logging helpers -----------------------------------------------------

    @property
    def output_std(self) -> torch.Tensor:
        """Mean action standard deviation from the last evaluation (for logging)."""
        if self._last_std is None:
            return torch.zeros(1)
        return self._last_std.mean().detach()

    # -- recurrent no-ops (parity with MLPModel) -----------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent state (no-op; non-recurrent model)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None``; non-recurrent model)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state (no-op; non-recurrent model)."""
        pass

    # -- export --------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-friendly deterministic actor for JIT export."""
        return _TorchFlashSACActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a deterministic actor wrapper for ONNX export."""
        return _OnnxFlashSACActor(self, verbose)


class FlashSACCriticModel(nn.Module):
    """Distributional double-Q critic with RSL-RL model surface (not exported)."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        action_dim: int,
        num_blocks: int = 2,
        hidden_dim: int = 256,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        num_qs: int = 2,
        obs_normalization: bool = False,
    ) -> None:
        """Initialize the critic model, resolving observation groups and the value support."""
        super().__init__()

        self.obs_groups, self.obs_dim = _get_obs_dim(obs, obs_groups, obs_set)
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v
        self.num_qs = num_qs

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.net = FlashSACDoubleCritic(
            num_blocks=num_blocks,
            input_dim=self.obs_dim + action_dim,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            num_qs=num_qs,
        )

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate the critic observation groups and normalize."""
        latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        return self.obs_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update the observation normalizer statistics (no-op if disabled)."""
        if self.obs_normalization:
            latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            self.obs_normalizer.update(latent)  # type: ignore[operator]

    def forward(
        self, obs: TensorDict, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return per-ensemble Q-values and per-bin log-probabilities for ``(obs, actions)``."""
        latent = self.get_latent(obs)
        return self.net(latent, actions, training)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Renormalize all weight-normalized submodules (call after each optimizer step)."""
        for module in self.net.modules():
            if hasattr(module, "normalize_parameters"):
                module.normalize_parameters()  # type: ignore[operator]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent state (no-op; non-recurrent model)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None``; non-recurrent model)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state (no-op; non-recurrent model)."""
        pass


class _TorchFlashSACActor(nn.Module):
    """TorchScript-friendly deterministic actor for JIT export."""

    def __init__(self, model: FlashSACActorModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.net = copy.deepcopy(model.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        mean, _ = self.net.get_mean_and_std(x, False)
        return torch.tanh(mean)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFlashSACActor(nn.Module):
    """Deterministic actor wrapper for ONNX export (pre-concatenated obs input)."""

    is_recurrent: bool = False

    def __init__(self, model: FlashSACActorModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.net = copy.deepcopy(model.net)
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        mean, _ = self.net.get_mean_and_std(x, False)
        return torch.tanh(mean)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
