from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Tuple, Union


# minimal interface of the environment
class VecEnv(ABC):
    """Abstract class for vectorized environment."""

    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor  # current episode duration
    extras: dict
    device: torch.device

    def __init__(
        self, observation_count, privileged_observation_count, device="cpu", environment_count=1, max_episode_length=-1
    ):
        """
        Args:
            observation_count (int): Number of observations per environment.
            privileged_observation_count (int): Number of privileged observations per environment.
            device (str): Device to use for the tensors.
            environment_count (int): Number of environments to run in parallel.
            max_episode_length (int): Maximum length of an episode. If -1, the episode length is not limited.
        """
        self.num_obs = observation_count
        self.num_privileged_obs = privileged_observation_count

        self.num_envs = environment_count
        self.max_episode_length = max_episode_length
        self.device = device

    @abstractmethod
    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return observations and extra information."""
        pass

    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        """Return privileged observations."""
        pass

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, privileged observations, rewards, dones and
                extra information (metrics).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]: Tuple containing the observations and privileged observations.
        """
        raise NotImplementedError
