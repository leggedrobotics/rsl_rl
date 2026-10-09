# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OffPolicyRunner (FlashSAC)."""

from __future__ import annotations

import copy
import os
import tempfile
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.env import VecEnv
from rsl_rl.runners import OffPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
PRIV_DIM = 5
NUM_ACTIONS = 4
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
        data = {
            "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
            "privileged": torch.randn(self.num_envs, PRIV_DIM, device=self.device),
        }
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_train_cfg() -> dict:
    """Return a minimal FlashSAC off-policy runner configuration."""
    return {
        "num_steps_per_env": 8,
        "updates_per_step": 1.0,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "multi_gpu": None,
        "torch_compile_mode": None,
        "algorithm": {
            "class_name": "FlashSAC",
            "gamma": 0.99,
            "n_step": 1,
            "learning_rate_init": 3e-4,
            "learning_rate_peak": 3e-4,
            "learning_rate_end": 1.5e-4,
            "learning_rate_warmup_steps": 0,
            "learning_rate_decay_steps": 1000,
            "critic_target_update_tau": 0.1,
            "num_bins": 21,
            "min_v": -5.0,
            "max_v": 5.0,
            "temp_initial_value": 0.01,
            "temp_target_sigma": 0.15,
            "temp_target_entropy": None,
            "actor_update_period": 2,
            "actor_bc_alpha": 0.0,
            "actor_noise_zeta_mu": 2.0,
            "actor_noise_zeta_max": 16,
            "normalize_reward": True,
            "normalized_g_max": 5.0,
            "use_amp": False,
        },
        "actor": {"class_name": "FlashSACActorModel", "num_blocks": 1, "hidden_dim": 16, "obs_normalization": False},
        "critic": {
            "class_name": "FlashSACCriticModel",
            "num_blocks": 1,
            "hidden_dim": 16,
            "num_bins": 21,
            "min_v": -5.0,
            "max_v": 5.0,
            "num_qs": 2,
            "obs_normalization": False,
        },
        "replay": {"capacity": 256, "min_length": 16, "sample_batch_size": 16},
    }


def _build_runner(log_dir: str | None = None) -> OffPolicyRunner:
    """Construct an OffPolicyRunner with a DummyEnv and minimal config."""
    return OffPolicyRunner(DummyEnv(), _make_train_cfg(), log_dir=log_dir, device="cpu")


class TestConstruction:
    """Tests for runner construction."""

    def test_runner_creates_algorithm(self) -> None:
        """Runner should instantiate a FlashSAC algorithm with actor and critic."""
        runner = _build_runner()
        assert runner.alg is not None
        assert runner.alg.actor is not None
        assert runner.alg.critic is not None

    def test_multi_gpu_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Construction raises when WORLD_SIZE > 1."""
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(NotImplementedError):
            _build_runner()


class TestLearnLoop:
    """Tests that the off-policy learn loop runs and updates parameters."""

    def test_learn_runs_without_error(self) -> None:
        """A short learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_critic_parameters(self) -> None:
        """Critic parameters should change after learning (buffer fills, updates run)."""
        runner = _build_runner()
        before = {n: p.clone() for n, p in runner.alg.critic.named_parameters()}
        runner.learn(num_learning_iterations=3)
        changed = any(not torch.equal(before[n], p) for n, p in runner.alg.critic.named_parameters())
        assert changed, "Critic parameters should have changed after learning"

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestSaveLoad:
    """Tests for checkpoint save and load."""

    def test_save_creates_file(self) -> None:
        """save() should create a checkpoint file with an iteration marker."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            data = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "iter" in data
            assert "actor_state_dict" in data

    def test_load_restores_actor_parameters(self) -> None:
        """Loading a checkpoint should restore actor parameters exactly."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())
            runner.learn(num_learning_iterations=2)
            runner.load(f.name)
            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Actor parameter '{key}' not restored after load"

    def test_actor_only_load(self) -> None:
        """Loading with load_cfg={'actor': True} should work (play path)."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            runner.load(f.name, load_cfg={"actor": True})


class TestInferenceAndExport:
    """Tests for inference policy and ONNX export."""

    def test_inference_policy_produces_bounded_actions(self) -> None:
        """The inference policy returns correctly shaped actions in [-1, 1]."""
        runner = _build_runner()
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert (actions.abs() <= 1.0 + 1e-6).all()

    @pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export(self) -> None:
        """Runner ONNX export should produce a policy.onnx file."""
        runner = _build_runner()
        with tempfile.TemporaryDirectory() as d:
            runner.export_policy_to_onnx(d, filename="policy.onnx")
            assert os.path.exists(os.path.join(d, "policy.onnx"))
