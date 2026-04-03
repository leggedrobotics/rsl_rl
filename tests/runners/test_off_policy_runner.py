# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OffPolicyRunner."""

from __future__ import annotations

import copy
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OffPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 3
MAX_EP_LEN = 50


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and rewards."""

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict(
            {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_train_cfg() -> dict:
    return {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "SAC",
            "batch_size": 16,
            "gradient_steps": 2,
            "learning_starts": 16,
            "replay_buffer_size": 2048,
            "learning_rate": 1e-3,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        },
    }


def _build_runner(log_dir: str | None = None) -> OffPolicyRunner:
    return OffPolicyRunner(DummyEnv(), _make_train_cfg(), log_dir=log_dir, device="cpu")


class TestOffPolicyRunner:
    """Runner construction and loop tests."""

    def test_runner_creates_algorithm(self) -> None:
        runner = _build_runner()
        assert runner.alg is not None

    def test_learn_runs_without_error(self) -> None:
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_parameters(self) -> None:
        runner = _build_runner()
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=4)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "Actor parameters should have changed after learning"

    def test_save_and_load(self) -> None:
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Parameter '{key}' not restored after load"