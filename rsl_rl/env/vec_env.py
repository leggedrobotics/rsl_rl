# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict


class VecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that
    the same type of action is applied to all environments and the same type of observation is returned from all
    environments.
    """

    num_envs: int
    """Number of environments."""

    num_actions: int
    """Number of actions."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length.

    The maximum episode length can be a scalar or a tensor. If it is a scalar, it is the same for all environments.
    If it is a tensor, it is the maximum episode length for each environment. This is useful for dynamic episode
    lengths.
    """

    episode_length_buf: torch.Tensor
    """Buffer for current episode lengths."""

    device: torch.device
    """Device to use."""

    cfg: dict | object
    """Configuration object."""

    """
    Operations.
    """

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return the current observations.

        Returns:
            observations (TensorDict): Observations from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Apply input action to the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
                observations (TensorDict): Observations from the environment.
                rewards (torch.Tensor): Rewards from the environment. Shape: (num_envs,)
                dones (torch.Tensor): Done flags from the environment. Shape: (num_envs,)
                extras (dict): Extra information from the environment.

        Observations:

            The observations TensorDict usually contains multiple observation groups. The `obs_groups`
            dictionary of the runner configuration specifies which observation groups are used for which
            purpose, i.e., it maps the available observation groups to observation sets. The observation sets
            (keys of the `obs_groups` dictionary) currently used by rsl_rl are:

            - "policy": Specified observation groups are used as input to the actor/student network.
            - "critic": Specified observation groups are used as input to the critic network.
            - "teacher": Specified observation groups are used as input to the teacher network.
            - "rnd_state": Specified observation groups are used as input to the RND network.

            Incomplete or incorrect configurations are handled in the `resolve_obs_groups()` function in
            `rsl_rl/utils/utils.py`.

        Extras:

            The extras dictionary includes metrics such as the episode reward, episode length, etc. The following
            dictionary keys are used by rsl_rl:

            - "time_outs" (torch.Tensor): Timeouts for the environments. These correspond to terminations that
               happen due to time limits and not due to the environment reaching a terminal state. This is useful
               for environments that have a fixed episode length.

            - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
               The key should be a string and start with "/" for namespacing. The value can be a scalar or a
               tensor. If it is a tensor, the mean of the tensor is used for logging.
        """
        raise NotImplementedError
