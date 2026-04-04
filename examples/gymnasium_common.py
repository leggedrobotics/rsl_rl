"""Shared utilities for running rsl_rl on Gymnasium classic benchmarks."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv


@dataclass
class RunArgs:
    """Common command-line arguments for examples."""

    env_id: str
    num_envs: int
    num_steps_per_env: int
    learning_iterations: int
    seed: int
    device: str
    log_dir: str | None


class GymnasiumVecEnv(VecEnv):
    """A simple VecEnv adapter for continuous Gymnasium environments."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 16,
        seed: int = 0,
        device: str = "cpu",
        render_mode: str | None = None,
    ) -> None:
        self.env_id = env_id
        self.num_envs = num_envs
        self.device = device
        self.cfg = {"env_id": env_id, "num_envs": num_envs, "seed": seed, "render_mode": render_mode}

        self.envs = [gym.make(env_id, render_mode=render_mode) for _ in range(num_envs)]
        action_space = self.envs[0].action_space
        obs_space = self.envs[0].observation_space
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError(f"Environment '{env_id}' does not have a continuous Box action space.")
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError(f"Environment '{env_id}' does not have a Box observation space.")

        self.num_actions = int(np.prod(action_space.shape))
        self._obs_dim = int(np.prod(obs_space.shape))
        self._action_low = action_space.low.reshape(-1)
        self._action_high = action_space.high.reshape(-1)

        max_episode_length = self.envs[0].spec.max_episode_steps if self.envs[0].spec is not None else None
        if max_episode_length is None:
            max_episode_length = 1000
        self.max_episode_length = int(max_episode_length)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_return_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._obs_buf = torch.zeros(self.num_envs, self._obs_dim, dtype=torch.float32, device=self.device)

        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            self._obs_buf[i] = self._flatten_obs(obs)

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._obs_buf.clone()}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().reshape(self.num_envs, self.num_actions)

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        episode_returns = []
        episode_lengths = []

        for i, env in enumerate(self.envs):
            action_i = np.clip(actions_np[i], self._action_low, self._action_high)
            obs, reward, terminated, truncated, _ = env.step(action_i)

            done = bool(terminated or truncated)
            reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
            rewards[i] = reward_t

            self._episode_return_buf[i] += reward_t
            self.episode_length_buf[i] += 1

            if done:
                dones[i] = 1.0
                if truncated:
                    time_outs[i] = 1.0
                episode_returns.append(self._episode_return_buf[i].item())
                episode_lengths.append(float(self.episode_length_buf[i].item()))

                obs, _ = env.reset()
                self._episode_return_buf[i] = 0.0
                self.episode_length_buf[i] = 0

            self._obs_buf[i] = self._flatten_obs(obs)

        extras: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {"time_outs": time_outs}
        if episode_returns:
            extras["log"] = {
                "/episode_return": torch.tensor(episode_returns, dtype=torch.float32, device=self.device).mean(),
                "/episode_length": torch.tensor(episode_lengths, dtype=torch.float32, device=self.device).mean(),
            }

        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @staticmethod
    def _flatten_obs(obs: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(-1))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach shared training arguments to a parser."""
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps-per-env", type=int, default=64)
    parser.add_argument("--learning-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default=None)
    return parser


def parse_common_args(description: str) -> RunArgs:
    """Parse common args and return a typed RunArgs object."""
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    args = parser.parse_args()
    return RunArgs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        num_steps_per_env=args.num_steps_per_env,
        learning_iterations=args.learning_iterations,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
    )


def make_log_dir(log_dir: str | None, default_run_name: str) -> str | None:
    """Create and return a logging directory if requested."""
    if log_dir is None:
        return None
    run_dir = os.path.join(log_dir, default_run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def set_seed(seed: int) -> None:
    """Set reproducible random seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ppo_train_cfg(num_steps_per_env: int, run_name: str | None = None) -> dict:
    """Create a compact PPO training config suitable for Gymnasium examples."""
    cfg = {
        "run_name": run_name,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 1000,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
        },
    }
    return cfg


def build_amp_ppo_train_cfg(
    num_steps_per_env: int,
    expert_observations: torch.Tensor | None = None,
    expert_observations_path: str | None = None,
    run_name: str | None = None,
) -> dict:
    """Create an AMP-PPO training config suitable for Gymnasium examples."""
    algorithm_cfg = {
        "class_name": "AMPPPO",
        "num_learning_epochs": 4,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.0,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "amp_reward_coef": 0.5,
        "discriminator_loss_coef": 1.0,
        "discriminator_learning_rate": 1e-3,
        "expert_batch_size": 256,
    }
    if expert_observations is not None:
        algorithm_cfg["expert_observations"] = expert_observations
    if expert_observations_path is not None:
        algorithm_cfg["expert_observations_path"] = expert_observations_path

    cfg = {
        "run_name": run_name,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 1000,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"], "amp": ["policy"]},
        "algorithm": algorithm_cfg,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
        },
        "discriminator": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
        },
    }
    return cfg


def build_sac_train_cfg(num_steps_per_env: int, run_name: str | None = None) -> dict:
    """Create a compact SAC training config suitable for Gymnasium examples."""
    cfg = {
        "run_name": run_name,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 1000,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "SAC",
            "gamma": 0.99,
            "tau": 0.005,
            "learning_rate": 3e-4,
            "auto_entropy_tuning": True,
            "replay_buffer_size": 100000,
            "batch_size": 256,
            "gradient_steps": 1,
            "learning_starts": 1000,
            "optimizer": "adam",
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "elu",
        },
    }
    return cfg
