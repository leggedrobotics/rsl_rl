# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Abstract class for vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that
    the same action is applied to all environments and the same observation is returned from all environments.

    All extra observations must be provided as a dictionary to "extras" in the step() method. Based on the
    configuration, the extra observations are used for different purposes. The following keys are used by the
    environment:

    - "observations" (dict[str, dict[str, torch.Tensor]]):
        Additional observations that are not used by the actor networks. The keys are the names of the observations
        and the values are the observations themselves. The following are reserved keys for the observations:

        - "critic": The observation is used as input to the critic network. Useful for asymmetric observation spaces.
        - "rnd_state": The observation is used as input to the RND network. Useful for random network distillation.

    - "time_outs" (torch.Tensor): Timeouts for the environments. These correspond to terminations that happen due to time limits and
      not due to the environment reaching a terminal state. This is useful for environments that have a fixed
      episode length.

    - "log" (dict[str, float | torch.Tensor]): Additional information for logging and debugging purposes.
      The key should be a string and start with "/" for namespacing. The value can be a scalar or a tensor.
      If it is a tensor, the mean of the tensor is used for logging.

      .. deprecated:: 2.0.0

        Use "log" in the extra information dictionary instead of the "episode" key.

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
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return the current observations.

        Returns:
            Tuple containing the observations and extras.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environment instances.

        Returns:
            Tuple containing the observations and extras.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply input action on the environment.

        The extra information is a dictionary. It includes metrics such as the episode reward, episode length,
        etc. Additional information can be stored in the dictionary such as observations for the critic network, etc.

        Args:
            actions: Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            A tuple containing the observations, rewards, dones and extra information (metrics).
        """
        raise NotImplementedError
