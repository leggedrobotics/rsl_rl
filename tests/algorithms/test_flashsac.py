# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FlashSAC algorithm."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict
from types import SimpleNamespace

import pytest

from rsl_rl.algorithms import FlashSAC

NUM_ENVS = 4
POLICY_DIM = 6
PRIV_DIM = 4
NUM_ACTIONS = 3


class _FakeEnv:
    """Minimal VecEnv-like object exposing what construct_algorithm needs."""

    def __init__(self, is_finite_horizon: bool = False) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.cfg = SimpleNamespace(is_finite_horizon=is_finite_horizon)


def _obs() -> TensorDict:
    """Create an observation TensorDict with 'policy' and 'privileged' groups."""
    return TensorDict(
        {"policy": torch.randn(NUM_ENVS, POLICY_DIM), "privileged": torch.randn(NUM_ENVS, PRIV_DIM)},
        batch_size=[NUM_ENVS],
    )


def _make_cfg(**alg_overrides: object) -> dict:
    """Build a full runner cfg dict for FlashSAC (fresh; construct_algorithm mutates it)."""
    algorithm: dict[str, object] = {
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
    }
    algorithm.update(alg_overrides)
    return {
        "algorithm": algorithm,
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
        "replay": {"capacity": 64, "min_length": 8, "sample_batch_size": 8},
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "multi_gpu": None,
        "torch_compile_mode": None,
    }


def _build(**alg_overrides: object) -> FlashSAC:
    """Construct a FlashSAC algorithm with small networks for testing."""
    return FlashSAC.construct_algorithm(_obs(), _FakeEnv(), _make_cfg(**alg_overrides), device="cpu")


def _fill_and_step(alg: FlashSAC, steps: int) -> None:
    """Drive act -> process_env_step for a number of steps to populate the buffer."""
    for _ in range(steps):
        obs = _obs()
        alg.act(obs, training=True)
        next_obs = _obs()
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(next_obs, rewards, dones, {"time_outs": torch.zeros(NUM_ENVS)})


class TestConstruction:
    """Tests for construct_algorithm and basic wiring."""

    def test_builds_and_target_matches_critic(self) -> None:
        """The target critic is initialized equal to the online critic."""
        alg = _build()
        for tp, cp in zip(alg.target_critic.parameters(), alg.critic.parameters()):
            assert torch.allclose(tp, cp)

    def test_target_entropy_autocomputed(self) -> None:
        """A None temp_target_entropy is auto-computed to a finite value."""
        alg = _build()
        assert alg.temp_target_entropy is not None
        assert torch.isfinite(torch.tensor(alg.temp_target_entropy))

    def test_compute_returns_is_noop(self) -> None:
        """compute_returns is a no-op for off-policy FlashSAC."""
        alg = _build()
        assert alg.compute_returns(_obs()) is None


class TestExplorationAndStorage:
    """Tests for the act/process_env_step rollout path."""

    def test_warmup_actions_bounded(self) -> None:
        """During warmup, actions are shaped correctly and lie in [-1, 1]."""
        alg = _build()
        actions = alg.act(_obs(), training=True)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert (actions.abs() <= 1.0 + 1e-6).all()

    def test_deterministic_eval_actions(self) -> None:
        """Evaluation actions are deterministic tanh(mean)."""
        alg = _build()
        obs = _obs()
        a1 = alg.act(obs, training=False)
        a2 = alg.act(obs, training=False)
        assert torch.allclose(a1, a2)

    def test_process_env_step_fills_buffer(self) -> None:
        """process_env_step adds transitions until the buffer can be sampled."""
        alg = _build()
        assert not alg.storage.can_sample()
        _fill_and_step(alg, 2)  # 2 * 4 envs = 8 == min_length
        assert alg.storage.can_sample()


class TestUpdate:
    """Tests for the update step."""

    def test_update_returns_finite_losses(self) -> None:
        """A full update returns finite critic (and, on actor steps, actor) losses."""
        alg = _build()
        _fill_and_step(alg, 4)
        loss = alg.update()
        assert "critic" in loss
        assert torch.isfinite(torch.tensor(loss["critic"]))
        # First update is at _update_step 0 -> actor update happens.
        assert "actor" in loss
        assert torch.isfinite(torch.tensor(loss["actor"]))

    def test_delayed_actor_update(self) -> None:
        """The actor is updated only every actor_update_period steps."""
        alg = _build(actor_update_period=2)
        _fill_and_step(alg, 4)
        first = alg.update()  # step 0 -> actor
        second = alg.update()  # step 1 -> no actor
        assert "actor" in first
        assert "actor" not in second

    def test_target_critic_drifts(self) -> None:
        """The EMA target critic moves after online-critic updates."""
        alg = _build()
        _fill_and_step(alg, 4)
        before = copy.deepcopy([p.detach().clone() for p in alg.target_critic.parameters()])
        for _ in range(3):
            alg.update()
        after = [p.detach() for p in alg.target_critic.parameters()]
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_update_before_ready_returns_empty(self) -> None:
        """update() before the buffer is fillable returns an empty dict."""
        alg = _build()
        assert alg.update() == {}


class TestSaveLoad:
    """Tests for checkpoint save/load."""

    def test_save_keys_and_actor_only_load(self) -> None:
        """save() returns the expected keys and an actor-only load works."""
        alg = _build()
        _fill_and_step(alg, 4)
        alg.update()
        state = alg.save()
        for key in ("actor_state_dict", "critic_state_dict", "target_critic_state_dict", "temperature_state_dict"):
            assert key in state

        fresh = _build()
        assert fresh.load(state, load_cfg={"actor": True}, strict=True) is False


class TestFailLoud:
    """Tests for fail-loud configuration and runtime validation."""

    def test_missing_section_raises(self) -> None:
        """A missing top-level cfg section raises KeyError."""
        cfg = _make_cfg()
        del cfg["replay"]
        with pytest.raises(KeyError):
            FlashSAC.construct_algorithm(_obs(), _FakeEnv(), cfg, device="cpu")

    def test_normalize_reward_false_raises(self) -> None:
        """normalize_reward=False is rejected (distributional support assumption)."""
        with pytest.raises(ValueError):
            _build(normalize_reward=False)

    def test_use_amp_true_raises(self) -> None:
        """use_amp=True is rejected in the eager-only v1."""
        with pytest.raises(NotImplementedError):
            _build(use_amp=True)

    def test_multi_gpu_raises(self) -> None:
        """A multi-GPU configuration is rejected."""
        cfg = _make_cfg()
        cfg["multi_gpu"] = {"global_rank": 0, "world_size": 2}
        with pytest.raises(NotImplementedError):
            FlashSAC.construct_algorithm(_obs(), _FakeEnv(), cfg, device="cpu")

    def test_missing_time_outs_infinite_horizon_raises(self) -> None:
        """An infinite-horizon env that omits time_outs raises KeyError in process_env_step."""
        alg = _build()
        alg.act(_obs(), training=True)
        with pytest.raises(KeyError):
            alg.process_env_step(_obs(), torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})

    def test_compile_mode_rejected(self) -> None:
        """A non-None torch_compile_mode is rejected (eager-only v1)."""
        cfg = _make_cfg()
        cfg["torch_compile_mode"] = "max-autotune"
        with pytest.raises(NotImplementedError):
            FlashSAC.construct_algorithm(_obs(), _FakeEnv(), cfg, device="cpu")
