from datetime import datetime
import gym
import torch
from typing import Any, Dict, Tuple, Union

from rsl_rl.env.vec_env import VecEnv


class GymEnv(VecEnv):
    """A vectorized environment wrapper for OpenAI Gym environments.

    This class wraps a single OpenAI Gym environment into a vectorized environment. It is assumed that the environment
    is a single agent environment. The environment is wrapped in a `gym.vector.SyncVectorEnv` environment, which
    allows for parallel execution of multiple environments.
    """

    def __init__(self, name, draw=False, draw_cb=None, draw_directory="videos/", gym_kwargs={}, **kwargs):
        """
        Args:
            name: The name of the OpenAI Gym environment.
            draw: Whether to record videos of the environment.
            draw_cb: A callback function that is called after each episode. The callback function is passed the episode
                number and the path to the video file. The callback function should return `True` if the video should
                be recorded and `False` otherwise.
            draw_directory: The directory in which to store the videos.
            gym_kwargs: Keyword arguments that are passed to the OpenAI Gym environment.
            **kwargs: Keyword arguments that are passed to the `VecEnv` constructor.
        """
        self._gym_kwargs = gym_kwargs

        env = gym.make(name, **self._gym_kwargs)

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 1
        assert isinstance(env.action_space, gym.spaces.Box)
        assert len(env.action_space.shape) == 1

        super().__init__(env.observation_space.shape[0], env.observation_space.shape[0], **kwargs)

        self.name = name
        self.draw_directory = draw_directory

        self.num_actions = env.action_space.shape[0]
        self._gym_venv = gym.vector.SyncVectorEnv(
            [lambda: gym.make(self.name, **self._gym_kwargs) for _ in range(self.num_envs)]
        )

        self._draw = False
        self._draw_cb = draw_cb if draw_cb is not None else lambda *args: True
        self._draw_uuid = None
        self.draw = draw

        self.reset()

    def close(self) -> None:
        self._gym_venv.close()

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.obs_buf, self.extras

    @property
    def draw(self) -> bool:
        return self._draw

    @draw.setter
    def draw(self, value: bool) -> None:
        if value != self._draw:
            if value:
                self._draw_uuid = datetime.now().strftime("%Y%m%d%H%M%S")
                env = gym.make(self.name, render_mode="rgb_array", **self._gym_kwargs)
                env = gym.wrappers.RecordVideo(
                    env,
                    f"{self.draw_directory}/{self._draw_uuid}/",
                    episode_trigger=lambda ep: (
                        self._draw_cb(ep - 1, f"{self.draw_directory}/{self._draw_uuid}/rl-video-episode-{ep-1}.mp4")
                        or True
                    )
                    if ep > 0
                    else False,
                )
            else:
                env = gym.make(self.name, render_mode=None, **self._gym_kwargs)

            self._gym_venv.envs[0] = env
            self._draw = value

        self.reset()

    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        self.obs_buf = torch.from_numpy(self._gym_venv.reset()[0]).float().to(self.device)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device).float()
        self.reset_buf = torch.zeros((self.num_envs,), device=self.device).float()
        self.extras = {"observations": {}, "time_outs": torch.zeros((self.num_envs,), device=self.device).float()}

        return self.obs_buf, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, rew, reset, term, _ = self._gym_venv.step(actions.cpu().numpy())

        self.obs_buf = torch.from_numpy(obs).float().to(self.device)
        self.rew_buf = torch.from_numpy(rew).float().to(self.device)
        self.reset_buf = torch.from_numpy(reset).float().to(self.device)
        self.extras = {
            "observations": {},
            "time_outs": torch.from_numpy(term).float().to(self.device).float().to(self.device),
        }

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def to(self, device: str) -> None:
        self.device = device

        self.obs_buf = self.obs_buf.to(device)
        self.rew_buf = self.rew_buf.to(device)
        self.reset_buf = self.reset_buf.to(device)
