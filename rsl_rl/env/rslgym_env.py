import torch
from typing import Any, Dict, Tuple, Union

from rsl_rl.env.vec_env import VecEnv


class RSLGymEnv(VecEnv):
    """A wrapper for using rsl_rl with the rslgym library."""

    def __init__(self, rslgym_env, **kwargs):
        self._rslgym_env = rslgym_env

        observation_count = self._rslgym_env.observation_space.shape[0]
        super().__init__(observation_count, observation_count, **kwargs)

        self.num_actions = self._rslgym_env.action_space.shape[0]

        self.obs_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self.extras = None

        self.reset()

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.obs_buf

    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        obs = self._rslgym_env.reset()

        self.obs_buf = torch.from_numpy(obs)
        self.extras = {"observations": {}, "time_outs": torch.zeros((self.num_envs,), device=self.device).float()}

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, reward, dones, info = self._rslgym_env.step(actions, True)

        self.obs_buf = torch.from_numpy(obs)
        self.rew_buf = torch.from_numpy(reward)
        self.reset_buf = torch.from_numpy(dones).float()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
