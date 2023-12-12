import math
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from typing import Any, Dict, Tuple, Union

from rsl_rl.env.vec_env import VecEnv


class PoleBalancing(VecEnv):
    """Custom pole balancing environment.

    This class implements a custom pole balancing environment. It demonstrates how to implement a custom `VecEnv`
    environment.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Keyword arguments that are passed to the `VecEnv` constructor.
        """
        super().__init__(2, 2, **kwargs)

        self.num_actions = 1

        self.gravity = 9.8
        self.length = 2.0
        self.dt = 0.1

        # Angle at which to fail the episode (15 deg)
        self.theta_threshold_radians = 15 * 2 * math.pi / 360

        # Max. angle at which to initialize the episode (5 deg)
        self.initial_max_position = 2 * 2 * math.pi / 360
        # Max. angular velocity at which to initialize the episode (1 deg/s)
        self.initial_max_velocity = 0.3 * 2 * math.pi / 360

        self.initial_position_factor = self.initial_max_position / 0.5
        self.initial_velocity_factor = self.initial_max_velocity / 0.5

        self.draw = False
        self.pushes = False

        self.reset()

    def get_observations(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        return self.obs_buf, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert actions.size() == (self.num_envs, 1)

        self.to(self.device)
        actions = actions.to(self.device)

        noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * 0.005).squeeze()
        if self.pushes and np.random.rand() < 0.05:
            noise *= 100.0
        actions = actions.clamp(min=-0.2, max=0.2).squeeze()
        gravity = torch.sin(self.state[:, 0]) * self.gravity / self.length
        angular_acceleration = gravity + actions + noise

        self.state[:, 1] = self.state[:, 1] + self.dt * angular_acceleration
        self.state[:, 0] = self.state[:, 0] + self.dt * self.state[:, 1]

        self.reset_buf = torch.zeros(self.num_envs)
        self.reset_buf[(torch.abs(self.state[:, 0]) > self.theta_threshold_radians).nonzero()] = 1.0
        reset_idx = self.reset_buf.nonzero()

        self.state[reset_idx, 0] = (
            torch.rand(reset_idx.size()[0], 1, device=self.device) - 0.5
        ) * self.initial_position_factor
        self.state[reset_idx, 1] = (
            torch.rand(reset_idx.size()[0], 1, device=self.device) - 0.5
        ) * self.initial_velocity_factor

        self.rew_buf = torch.ones(self.num_envs, device=self.device)
        self.rew_buf[reset_idx] = -1.0
        self.rew_buf = self.rew_buf - actions.abs()
        self.rew_buf = self.rew_buf - self.state[:, 0].abs()

        self._update_obs()

        if self.draw:
            self._debug_draw(actions)

        self.to(self.device)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _update_obs(self):
        self.obs_buf = self.state
        self.extras = {"observations": {}, "time_outs": torch.zeros_like(self.rew_buf)}

    def reset(self) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        self.state = torch.zeros(self.num_envs, 2, device=self.device)
        self.state[:, 0] = (torch.rand(self.num_envs) - 0.5) * self.initial_position_factor
        self.state[:, 1] = (torch.rand(self.num_envs) - 0.5) * self.initial_velocity_factor
        self.rew_buf = torch.zeros(self.num_envs)
        self.reset_buf = torch.zeros(self.num_envs)
        self.extras = {}

        self._update_obs()

        return self.obs_buf, self.extras

    def to(self, device):
        self.device = device

        self.obs_buf = self.obs_buf.to(device)
        self.rew_buf = self.rew_buf.to(device)
        self.reset_buf = self.reset_buf.to(device)
        self.state = self.state.to(device)

    def _debug_draw(self, actions):
        if not hasattr(self, "_visuals"):
            self._visuals = {"x": [0], "pos": [], "act": [], "done": []}
            plt.gca().figure.show()
        else:
            self._visuals["x"].append(self._visuals["x"][-1] + 1)

        self._visuals["pos"].append(self.obs_buf[0, 0].cpu().item())
        self._visuals["done"].append(self.reset_buf[0].cpu().item())
        self._visuals["act"].append(actions.squeeze()[0].cpu().item())

        plt.cla()
        plt.plot(self._visuals["x"][-100:], self._visuals["act"][-100:], color="green")
        plt.plot(self._visuals["x"][-100:], self._visuals["pos"][-100:], color="blue")
        plt.plot(self._visuals["x"][-100:], self._visuals["done"][-100:], color="red")
        plt.draw()
        plt.gca().figure.canvas.flush_events()
        time.sleep(0.0001)
