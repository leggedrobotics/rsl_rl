#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Abstract class for vectorized environment.

    The vectorized environment is a collection of environments that are synchronized. This means that
    the same action is applied to all environments and the same observation is returned from all environments.

    All extra observations must be provided as a dictionary to "extras" in the step() method. Based on the
    configuration, the extra observations are used for different purposes. The following keys are reserved
    in the "observations" dictionary (if they are present):

    - "critic": The observation is used as input to the critic network. Useful for asymmetric observation spaces.
    """

    num_envs: int
    """Number of environments."""
    num_obs: int
    """Number of observations."""
    num_privileged_obs: int
    """Number of privileged observations."""
    num_actions: int
    """Number of actions."""
    max_episode_length: int
    """Maximum episode length."""
    privileged_obs_buf: torch.Tensor
    """Buffer for privileged observations."""
    obs_buf: torch.Tensor
    """Buffer for observations."""
    rew_buf: torch.Tensor
    """Buffer for rewards."""
    reset_buf: torch.Tensor
    """Buffer for resets."""
    episode_length_buf: torch.Tensor  # current episode duration
    """Buffer for current episode lengths."""
    extras: dict
    """Extra information (metrics).

    Extra information is stored in a dictionary. This includes metrics such as the episode reward, episode length,
    etc. Additional information can be stored in the dictionary such as observations for the critic network, etc.
    """
    device: torch.device
    """Device to use."""

    """
    Operations.
    """

    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return the current observations.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, rewards, dones and extra information (metrics).
        """
        raise NotImplementedError
