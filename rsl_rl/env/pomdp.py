import torch
from typing import Any, Dict, Tuple, Union

from rsl_rl.env.gym_env import GymEnv


class GymPOMDP(GymEnv):
    """A vectorized POMDP environment wrapper for OpenAI Gym environments.

    This environment allows for the modification of the observation space of an OpenAI Gym environment. The modified
    observation space is a subset of the original observation space.
    """

    _reduced_observation_count: int = None

    def __init__(self, name: str, **kwargs):
        assert self._reduced_observation_count is not None

        super().__init__(name=name, **kwargs)

        self.num_obs = self._reduced_observation_count
        self.num_privileged_obs = self._reduced_observation_count

    def _process_observations(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reduces observation space from original observation space to modified observation space.

        Args:
            obs (torch.Tensor): Original observations.
        Returns:
            The modified observations as a torch.Tensor of shape (obs.shape[0], self.num_obs).
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        obs, _ = super().reset(*args, **kwargs)

        self.obs_buf = self._process_observations(obs)

        return self.obs_buf, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, _, _, _ = super().step(actions)

        self.obs_buf = self._process_observations(obs)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras


class BipedalWalkerP(GymPOMDP):
    """
    Original observation space (24 values):
        [
            hull angle,
            hull angular velocity,
            horizontal velocity,
            vertical velocity,
            joint 1 angle,
            joint 1 speed,
            joint 2 angle,
            joint 2 speed,
            leg 1 ground contact,
            joint 3 angle,
            joint 3 speed,
            joint 4 angle,
            joint 4 speed,
            leg 2 ground contact,
            lidar (10 values),
        ]
    Modified observation space (15 values):
        [
            hull angle,
            joint 1 angle,
            joint 2 angle,
            joint 3 angle,
            joint 4 angle,
            lidar (10 values),
        ]
    """

    _reduced_observation_count: int = 15

    def __init__(self, **kwargs):
        super().__init__(name="BipedalWalker-v3", **kwargs)

    def _process_observations(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reduces observation space from original observation space to modified observation space."""
        reduced_obs = torch.zeros(obs.shape[0], self._reduced_observation_count, device=self.device)
        reduced_obs[:, 0] = obs[:, 0]
        reduced_obs[:, 1] = obs[:, 4]
        reduced_obs[:, 2] = obs[:, 6]
        reduced_obs[:, 3] = obs[:, 9]
        reduced_obs[:, 4] = obs[:, 11]
        reduced_obs[:, 5:] = obs[:, 14:]

        return reduced_obs
