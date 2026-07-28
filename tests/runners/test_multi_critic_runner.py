# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MultiCriticOnPolicyRunner used with MultiCriticPPO.

MultiCriticOnPolicyRunner is generic over the algorithm class (it resolves `cfg["algorithm"]["class_name"]` and
calls `alg_class.construct_algorithm(...)`), so no runner subclass is needed for multi-critic PPO — we
only need a config that sets `class_name: "MultiCriticPPO"` and `num_critics: N`. These tests exercise
that full config-driven path end to end (resolve_obs_groups / resolve_rnd_config / resolve_symmetry_config
plus the real learn/save/load loop), which is a stronger integration check than directly constructing
`MultiCriticPPO` the way tests/algorithms/test_multi_critic_ppo.py does.
"""

from __future__ import annotations

import copy
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MultiCriticPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners import MultiCriticOnPolicyRunner

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 50


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and one reward per critic."""

    def __init__(
        self,
        num_critics: int,
        device: str = "cpu",
    ) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.num_critics = num_critics
        self.max_episode_length = MAX_EP_LEN

        self.episode_length_buf = torch.zeros(
            NUM_ENVS,
            dtype=torch.long,
            device=device,
        )

        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:
        data = {
            "policy": torch.randn(
                self.num_envs,
                OBS_DIM,
                device=self.device,
            )
        }

        return TensorDict(
            data,
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
        TensorDict,
        tuple[torch.Tensor, ...],
        torch.Tensor,
        dict,
    ]:
        self.episode_length_buf += 1

        dones = (
            self.episode_length_buf >= self.max_episode_length
        ).float()

        self.episode_length_buf[dones.bool()] = 0

        obs = self.get_observations()

        rewards = tuple(
            torch.randn(
                self.num_envs,
                device=self.device,
            )
            for _ in range(self.num_critics)
        )

        extras = {
            "time_outs": torch.zeros(
                self.num_envs,
                device=self.device,
            )
        }

        return obs, rewards, dones, extras


def _make_multi_critic_train_cfg(model_type: str = "mlp", num_critics: int = 2) -> dict:
    """Return a minimal MultiCriticPPO training configuration.

    Mirrors `_make_train_cfg` in tests/runners/test_runner.py, but sets `algorithm.class_name` to
    ``"MultiCriticPPO"`` and adds `algorithm.num_critics`, which `MultiCriticPPO.construct_algorithm`
    pops off to build that many critics from the single `cfg["critic"]` spec.

    Args:
        model_type: One of ``"mlp"`` or ``"rnn"``.
        num_critics: Number of critics for MultiCriticPPO to construct.
    """
    cfg: dict = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {
            "actor": ["policy"],
            **{
                f"critic_{i}": ["policy"]
                for i in range(num_critics)
            },
        },
        "algorithm": {
            "class_name": "MultiCriticPPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "num_critics": num_critics,
        },
    }
    if model_type == "rnn":
        cfg["actor"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        cfg["critic"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
        }
    else:
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        }
    return cfg


def _build_multi_critic_runner(
    log_dir: str | None = None,
    model_type: str = "mlp",
    num_critics: int = 2,
) -> MultiCriticOnPolicyRunner:
    env = DummyEnv(num_critics=num_critics)

    cfg = _make_multi_critic_train_cfg(
        model_type,
        num_critics,
    )

    return MultiCriticOnPolicyRunner(
        env,
        cfg,
        log_dir=log_dir,
        device="cpu",
    )


class TestMultiCriticRunnerConstruction:
    """Tests for constructing the runner with a MultiCriticPPO algorithm."""

    def test_runner_creates_multi_critic_algorithm(self) -> None:
        """Runner should instantiate a MultiCriticPPO with an actor and the requested number of critics."""
        runner = _build_multi_critic_runner(num_critics=3)
        assert isinstance(runner.alg, MultiCriticPPO)
        assert runner.alg.actor is not None
        assert len(runner.alg.critics) == 3

    def test_runner_sets_initial_iteration(self) -> None:
        """Initial learning iteration should be zero."""
        runner = _build_multi_critic_runner()
        assert runner.current_learning_iteration == 0


class TestMultiCriticLearnLoop:
    """Tests that the learn loop runs and updates parameters for every critic."""

    def test_learn_runs_without_error(self) -> None:
        """A short learn call should complete without raising."""
        runner = _build_multi_critic_runner(num_critics=3)
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_actor_parameters(self) -> None:
        """Actor parameters should change after a learning iteration."""
        runner = _build_multi_critic_runner(num_critics=3)
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "Actor parameters should have changed after learning"

    def test_learn_updates_every_critic(self) -> None:
        """Every critic in the list — not just the first — should receive gradient updates."""
        runner = _build_multi_critic_runner(num_critics=3)
        params_before = [{n: p.clone() for n, p in critic.named_parameters()} for critic in runner.alg.critics]

        runner.learn(num_learning_iterations=2)

        for c_idx, (before, critic) in enumerate(zip(params_before, runner.alg.critics)):
            changed = any(not torch.equal(before[n], p) for n, p in critic.named_parameters())
            assert changed, f"Critic {c_idx} parameters should have changed after learning"

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed iterations."""
        runner = _build_multi_critic_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestMultiCriticSaveLoad:
    """Tests for checkpoint save and load with multiple critics."""

    def test_save_creates_file_with_all_critics(self) -> None:
        """save() should create a checkpoint containing every critic's state dict."""
        runner = _build_multi_critic_runner(num_critics=3)
        runner.learn(num_learning_iterations=1)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            data = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "iter" in data
            assert len(data["critic_state_dict"]) == 3

    def test_load_restores_every_critic(self) -> None:
        """Loading a checkpoint should restore every critic's parameters exactly, not just one."""
        runner = _build_multi_critic_runner(num_critics=3)
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_critics = [copy.deepcopy(critic.state_dict()) for critic in runner.alg.critics]

            runner.learn(num_learning_iterations=2)
            any_changed = any(
                not all(torch.equal(saved[k], v) for k, v in critic.state_dict().items())
                for saved, critic in zip(saved_critics, runner.alg.critics)
            )
            assert any_changed, "Critic parameters should have changed after additional training"

            runner.load(f.name)
            for c_idx, (saved, critic) in enumerate(zip(saved_critics, runner.alg.critics)):
                for key, param in critic.state_dict().items():
                    assert torch.equal(saved[key], param), f"Critic {c_idx} parameter '{key}' not restored after load"

    def test_load_restores_actor(self) -> None:
        """Loading a checkpoint should also restore the (single) actor's parameters."""
        runner = _build_multi_critic_runner(num_critics=2)
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Actor parameter '{key}' not restored after load"

    def test_load_restores_iteration(self) -> None:
        """Loading a checkpoint should restore the iteration counter."""
        runner = _build_multi_critic_runner(num_critics=2)
        runner.learn(num_learning_iterations=3)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_iter = runner.current_learning_iteration

            runner.learn(num_learning_iterations=2)
            assert runner.current_learning_iteration != saved_iter

            runner.load(f.name)
            assert runner.current_learning_iteration == saved_iter


class TestMultiCriticInferencePolicy:
    """Tests for get_inference_policy with a MultiCriticPPO-backed runner."""

    def test_inference_policy_returns_callable(self) -> None:
        """get_inference_policy should return a callable model."""
        runner = _build_multi_critic_runner()
        policy = runner.get_inference_policy()
        assert callable(policy)

    def test_inference_policy_produces_actions(self) -> None:
        """The inference policy should return a tensor with the correct action shape."""
        runner = _build_multi_critic_runner(num_critics=3)
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)


class TestMultiCriticDeterministicTraining:
    """Two seeded training runs should produce identical actor parameters."""

    @staticmethod
    def _seeded_train(seed: int, num_critics: int = 2, model_type: str = "mlp") -> dict[str, torch.Tensor]:
        """Run a short training loop with a fixed seed and return the actor's state_dict."""
        torch.manual_seed(seed)
        runner = _build_multi_critic_runner(model_type=model_type, num_critics=num_critics)
        runner.learn(num_learning_iterations=3)
        return {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}

    def test_reproducibility(self) -> None:
        """Two runs with the same seed should yield identical actor parameters."""
        run_a = self._seeded_train(seed=42)
        run_b = self._seeded_train(seed=42)
        for key in run_a:
            assert torch.equal(run_a[key], run_b[key]), f"Param '{key}' differs between seeded runs"

    def test_different_seeds_diverge(self) -> None:
        """Different seeds should produce different parameters."""
        run_a = self._seeded_train(seed=42)
        run_b = self._seeded_train(seed=99)
        any_different = any(not torch.equal(run_a[k], run_b[k]) for k in run_a)
        assert any_different, "Different seeds should produce different parameters"


class TestMultiCriticRNNRunner:
    """Full learn loop with a recurrent actor/critics, via the real config-driven RNNModel path."""

    def test_rnn_learn_runs_without_error(self) -> None:
        """A short learn call with RNNModel actor/critics should complete without raising."""
        runner = _build_multi_critic_runner(model_type="rnn", num_critics=2)
        runner.learn(num_learning_iterations=2)

    def test_rnn_learn_updates_every_critic(self) -> None:
        """Every recurrent critic should receive gradient updates."""
        runner = _build_multi_critic_runner(model_type="rnn", num_critics=2)
        params_before = [{n: p.clone() for n, p in critic.named_parameters()} for critic in runner.alg.critics]

        runner.learn(num_learning_iterations=2)

        for c_idx, (before, critic) in enumerate(zip(params_before, runner.alg.critics)):
            changed = any(not torch.equal(before[n], p) for n, p in critic.named_parameters())
            assert changed, f"Recurrent critic {c_idx} parameters should have changed after learning"

    def test_rnn_inference_produces_actions(self) -> None:
        """Inference policy from an RNN-based multi-critic runner should return correct action shape."""
        runner = _build_multi_critic_runner(model_type="rnn", num_critics=2)
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_rnn_save_load_restores_every_critic(self) -> None:
        """Save/load should preserve every recurrent critic's parameters, including RNN weights."""
        runner = _build_multi_critic_runner(model_type="rnn", num_critics=2)
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_critics = [copy.deepcopy(critic.state_dict()) for critic in runner.alg.critics]

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for c_idx, (saved, critic) in enumerate(zip(saved_critics, runner.alg.critics)):
                for key, param in critic.state_dict().items():
                    assert torch.equal(saved[key], param), (
                        f"RNN critic {c_idx} parameter '{key}' not restored after load"
                    )