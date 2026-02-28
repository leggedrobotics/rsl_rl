# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OnPolicyRunner."""

from __future__ import annotations

import copy
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 50
IMG_C, IMG_H, IMG_W = 1, 16, 16


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and rewards."""

    def __init__(self, device: str = "cpu", include_image: bool = False) -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}
        self._include_image = include_image

    def get_observations(self) -> TensorDict:  # noqa: D102
        data: dict = {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)}
        if self._include_image:
            data["image"] = torch.randn(self.num_envs, IMG_C, IMG_H, IMG_W, device=self.device)
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_train_cfg(model_type: str = "mlp") -> dict:
    """Return a minimal training configuration for PPO.

    Args:
        model_type: One of ``"mlp"``, ``"rnn"``, or ``"cnn"``.
    """
    cfg: dict = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
        },
    }
    if model_type == "rnn":
        cfg["actor"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }
        cfg["critic"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
        }
    elif model_type == "cnn":
        cfg["obs_groups"] = {
            "actor": ["policy", "image"],
            "critic": ["policy", "image"],
        }
        cnn_cfg = {
            "output_channels": [4],
            "kernel_size": 3,
            "stride": 2,
        }
        cfg["actor"] = {
            "class_name": "CNNModel",
            "hidden_dims": [32],
            "activation": "elu",
            "cnn_cfg": cnn_cfg,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }
        cfg["critic"] = {
            "class_name": "CNNModel",
            "hidden_dims": [32],
            "activation": "elu",
            "cnn_cfg": cnn_cfg,
        }
    else:
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        }
    return cfg


def _build_runner(log_dir: str | None = None, model_type: str = "mlp") -> OnPolicyRunner:
    """Construct a runner with a DummyEnv and minimal config."""
    env = DummyEnv(include_image=(model_type == "cnn"))
    cfg = _make_train_cfg(model_type)
    return OnPolicyRunner(env, cfg, log_dir=log_dir, device="cpu")


class TestRunnerConstruction:
    """Tests for constructing the runner and its components."""

    def test_runner_creates_algorithm(self) -> None:
        """Runner should instantiate a PPO algorithm with actor and critic."""
        runner = _build_runner()
        assert runner.alg is not None
        assert runner.alg.actor is not None
        assert runner.alg.critic is not None

    def test_runner_sets_initial_iteration(self) -> None:
        """Initial learning iteration should be zero."""
        runner = _build_runner()
        assert runner.current_learning_iteration == 0


class TestLearnLoop:
    """Tests that the learn loop runs and updates parameters."""

    def test_learn_runs_without_error(self) -> None:
        """A short learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_parameters(self) -> None:
        """Actor parameters should change after a learning iteration."""
        runner = _build_runner()
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "Actor parameters should have changed after learning"

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestSaveLoad:
    """Tests for checkpoint save and load."""

    def test_save_creates_file(self) -> None:
        """save() should create a checkpoint file at the given path."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            data = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "iter" in data

    def test_load_restores_parameters(self) -> None:
        """Loading a checkpoint should restore model parameters exactly."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            assert not all(torch.equal(saved_actor[k], v) for k, v in runner.alg.actor.state_dict().items()), (
                "Parameters should have changed after additional training"
            )

            runner.load(f.name)
            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Parameter '{key}' not restored after load"

    def test_load_restores_iteration(self) -> None:
        """Loading a checkpoint should restore the iteration counter."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_iter = runner.current_learning_iteration

            runner.learn(num_learning_iterations=2)
            assert runner.current_learning_iteration != saved_iter

            runner.load(f.name)
            assert runner.current_learning_iteration == saved_iter

    def test_load_restores_normalization_stats(self) -> None:
        """Running-mean stats should be identical after save and load."""
        cfg = _make_train_cfg("mlp")
        cfg["actor"]["obs_normalization"] = True
        cfg["critic"]["obs_normalization"] = True

        runner = OnPolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_state = copy.deepcopy(runner.alg.actor.state_dict())

            cfg2 = _make_train_cfg("mlp")
            cfg2["actor"]["obs_normalization"] = True
            cfg2["critic"]["obs_normalization"] = True
            runner2 = OnPolicyRunner(DummyEnv(), cfg2, log_dir=None, device="cpu")
            runner2.load(f.name)

            for key, param in runner2.alg.actor.state_dict().items():
                assert torch.equal(saved_state[key], param), f"Normalization stat '{key}' not restored after load"


class TestInferencePolicy:
    """Tests for get_inference_policy and the returned callable."""

    def test_inference_policy_returns_callable(self) -> None:
        """get_inference_policy should return a callable model."""
        runner = _build_runner()
        policy = runner.get_inference_policy()
        assert callable(policy)

    def test_inference_policy_produces_actions(self) -> None:
        """The inference policy should return a tensor with the correct action shape."""
        runner = _build_runner()
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_inference_loop(self) -> None:
        """Simulate a replay loop: step the env with policy outputs for several steps."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        policy = runner.get_inference_policy()

        obs = runner.env.get_observations()
        for _ in range(5):
            actions = policy(obs)
            obs, rewards, _dones, _extras = runner.env.step(actions)
            assert rewards.shape == (NUM_ENVS,)


class TestDeterministicTraining:
    """Two seeded training runs should produce identical results."""

    @staticmethod
    def _seeded_train(seed: int, model_type: str = "mlp") -> dict[str, torch.Tensor]:
        """Run a short training loop with a fixed seed and return actor state_dict."""
        torch.manual_seed(seed)
        runner = _build_runner(model_type=model_type)
        runner.learn(num_learning_iterations=3)
        return {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}

    def test_mlp_reproducibility(self) -> None:
        """Two MLP training runs with the same seed should yield identical parameters."""
        run_a = self._seeded_train(seed=42, model_type="mlp")
        run_b = self._seeded_train(seed=42, model_type="mlp")
        for key in run_a:
            assert torch.equal(run_a[key], run_b[key]), f"MLP param '{key}' differs between seeded runs"

    def test_rnn_reproducibility(self) -> None:
        """Two RNN training runs with the same seed should yield identical parameters."""
        run_a = self._seeded_train(seed=42, model_type="rnn")
        run_b = self._seeded_train(seed=42, model_type="rnn")
        for key in run_a:
            assert torch.equal(run_a[key], run_b[key]), f"RNN param '{key}' differs between seeded runs"

    def test_different_seeds_diverge(self) -> None:
        """Different seeds should produce different parameters."""
        run_a = self._seeded_train(seed=42)
        run_b = self._seeded_train(seed=99)
        any_different = any(not torch.equal(run_a[k], run_b[k]) for k in run_a)
        assert any_different, "Different seeds should produce different parameters"


class TestRNNRunner:
    """Tests that the full learn loop works with an RNN-based actor/critic."""

    def test_rnn_learn_runs_without_error(self) -> None:
        """A short learn call with RNN models should complete without raising."""
        runner = _build_runner(model_type="rnn")
        runner.learn(num_learning_iterations=2)

    def test_rnn_learn_updates_parameters(self) -> None:
        """RNN actor parameters should change after learning."""
        runner = _build_runner(model_type="rnn")
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "RNN actor parameters should have changed after learning"

    def test_rnn_inference_produces_actions(self) -> None:
        """Inference policy from an RNN runner should return correct action shape."""
        runner = _build_runner(model_type="rnn")
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_rnn_save_load_restores_parameters(self) -> None:
        """Save/load should preserve RNN model parameters (including hidden state shapes)."""
        runner = _build_runner(model_type="rnn")
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"RNN parameter '{key}' not restored after load"


class TestCNNRunner:
    """Tests that the full learn loop works with a CNN-based actor/critic."""

    def test_cnn_learn_runs_without_error(self) -> None:
        """A short learn call with CNN models should complete without raising."""
        runner = _build_runner(model_type="cnn")
        runner.learn(num_learning_iterations=2)

    def test_cnn_learn_updates_parameters(self) -> None:
        """CNN actor parameters should change after learning."""
        runner = _build_runner(model_type="cnn")
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "CNN actor parameters should have changed after learning"

    def test_cnn_inference_produces_actions(self) -> None:
        """Inference policy from a CNN runner should return correct action shape."""
        runner = _build_runner(model_type="cnn")
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_cnn_save_load_restores_parameters(self) -> None:
        """Save/load should preserve CNN model parameters."""
        runner = _build_runner(model_type="cnn")
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"CNN parameter '{key}' not restored after load"
