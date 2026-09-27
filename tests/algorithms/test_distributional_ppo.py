# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DistributionalPPO algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from types import SimpleNamespace

import pytest

from rsl_rl.algorithms import DistributionalPPO
from rsl_rl.algorithms.distributional_ppo import quantile_huber_loss
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4
NUM_QUANTILES = 8


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, num_outputs: int = 1, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", num_outputs, **defaults)


def _build_distributional_ppo(
    num_quantiles: int = NUM_QUANTILES, **overrides: object
) -> tuple[DistributionalPPO, TensorDict, RolloutStorage]:
    """Build a DistributionalPPO instance with a quantile critic for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
    # Critic outputs num_quantiles instead of scalar 1
    critic = _make_critic(obs, obs_groups, num_outputs=num_quantiles)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults: dict[str, object] = dict(num_quantiles=num_quantiles, schedule="fixed")
    defaults.update(overrides)
    ppo = DistributionalPPO(actor, critic, storage, **defaults)  # type: ignore[arg-type]
    return ppo, obs, storage


def _rollout(ppo: DistributionalPPO, obs: TensorDict) -> None:
    """Collect a full rollout of transitions and compute returns."""
    for _ in range(NUM_STEPS):
        ppo.act(obs)
        ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
    ppo.compute_returns(obs)


class TestQuantileHuberLoss:
    """Tests for the standalone ``quantile_huber_loss`` function."""

    def test_zero_error_gives_zero_loss(self) -> None:
        """When predictions equal targets across all quantiles, loss is zero."""
        targets = torch.tensor([[2.0], [3.0]])
        quantiles = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        loss = quantile_huber_loss(quantiles, targets)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_is_non_negative(self) -> None:
        """Loss should be non-negative for arbitrary inputs."""
        quantiles = torch.randn(10, NUM_QUANTILES)
        targets = torch.randn(10, 1)
        loss = quantile_huber_loss(quantiles, targets)
        assert loss.item() >= 0.0

    def test_returns_scalar(self) -> None:
        """Loss is reduced to a scalar regardless of batch shape."""
        quantiles = torch.randn(10, NUM_QUANTILES)
        targets = torch.randn(10, 1)
        loss = quantile_huber_loss(quantiles, targets)
        assert loss.shape == ()

    def test_accepts_1d_targets(self) -> None:
        """1-D targets of shape (B,) are broadcast identically to (B, 1) targets."""
        quantiles = torch.randn(10, NUM_QUANTILES)
        targets = torch.randn(10)
        loss_1d = quantile_huber_loss(quantiles, targets)
        loss_2d = quantile_huber_loss(quantiles, targets.unsqueeze(-1))
        assert torch.allclose(loss_1d, loss_2d)

    def test_hand_computed_single_quantile(self) -> None:
        """With one quantile (tau=0.5) and small error, loss equals 0.5 * 0.5 * err^2."""
        # tau = 0.5, error = 0.4 (< kappa=1.0), so huber = 0.5 * 0.4^2 = 0.08
        # check weight = |0.5 - 0| = 0.5 (error > 0), loss = 0.5 * 0.08 = 0.04
        quantiles = torch.tensor([[1.0]])
        targets = torch.tensor([[1.4]])
        loss = quantile_huber_loss(quantiles, targets, kappa=1.0)
        assert torch.allclose(loss, torch.tensor(0.04), atol=1e-6)

    def test_linear_regime_beyond_kappa(self) -> None:
        """Errors larger than kappa use the linear Huber branch."""
        # tau = 0.5, error = 3.0 (> kappa=1.0), so huber = 1.0 * (3.0 - 0.5) = 2.5
        # check weight = 0.5, loss = 0.5 * 2.5 = 1.25
        quantiles = torch.tensor([[0.0]])
        targets = torch.tensor([[3.0]])
        loss = quantile_huber_loss(quantiles, targets, kappa=1.0)
        assert torch.allclose(loss, torch.tensor(1.25), atol=1e-6)

    def test_asymmetric_weighting(self) -> None:
        """Over-estimating a low quantile is penalized more than under-estimating it by the same amount."""
        # With 2 quantiles tau = [0.25, 0.75]. Make only the first (low) quantile wrong by setting the
        # second quantile exactly equal to the target so it contributes zero loss.
        # Under-estimate (target > pred): weight = |0.25 - 0| = 0.25, huber(0.5) = 0.125
        #   loss = mean([0.25 * 0.125, 0]) = 0.015625
        # Over-estimate  (target < pred): weight = |0.25 - 1| = 0.75, huber(0.5) = 0.125
        #   loss = mean([0.75 * 0.125, 0]) = 0.046875
        loss_under = quantile_huber_loss(torch.tensor([[0.0, 0.5]]), torch.tensor([[0.5]]))
        loss_over = quantile_huber_loss(torch.tensor([[0.0, -0.5]]), torch.tensor([[-0.5]]))
        assert torch.allclose(loss_under, torch.tensor(0.015625), atol=1e-6)
        assert torch.allclose(loss_over, torch.tensor(0.046875), atol=1e-6)
        assert loss_over > loss_under

    def test_gradient_flows(self) -> None:
        """Loss is differentiable with respect to predicted quantiles."""
        quantiles = torch.randn(10, NUM_QUANTILES, requires_grad=True)
        targets = torch.randn(10, 1)
        loss = quantile_huber_loss(quantiles, targets)
        loss.backward()
        assert quantiles.grad is not None
        assert torch.isfinite(quantiles.grad).all()


class TestDistributionalCriticLoss:
    """Tests verifying DistributionalPPO.compute_critic_loss dispatch."""

    def test_zero_error_gives_zero_loss(self) -> None:
        """When predictions equal targets across all quantiles, loss is zero."""
        ppo, _, _ = _build_distributional_ppo(num_quantiles=3)
        targets = torch.tensor([[2.0], [3.0]])
        quantiles = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        loss = ppo.compute_critic_loss(quantiles, targets, quantiles)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_is_non_negative(self) -> None:
        """Loss should be non-negative for arbitrary inputs."""
        ppo, _, _ = _build_distributional_ppo()
        quantiles = torch.randn(10, NUM_QUANTILES)
        targets = torch.randn(10, 1)
        loss = ppo.compute_critic_loss(quantiles, targets, quantiles)
        assert loss.item() >= 0.0

    def test_matches_standalone_function(self) -> None:
        """compute_critic_loss delegates to quantile_huber_loss with the configured kappa."""
        kappa = 0.5
        ppo, _, _ = _build_distributional_ppo(quantile_huber_kappa=kappa)
        quantiles = torch.randn(10, NUM_QUANTILES)
        targets = torch.randn(10, 1)
        loss = ppo.compute_critic_loss(quantiles, targets, quantiles)
        expected = quantile_huber_loss(quantiles, targets, kappa=kappa)
        assert torch.allclose(loss, expected)

    def test_falls_back_to_ppo_loss_when_quantiles_disabled(self) -> None:
        """With num_quantiles=0, the base PPO MSE loss is used."""
        ppo, _, _ = _build_distributional_ppo(num_quantiles=1, use_clipped_value_loss=False)
        ppo.num_quantiles = 0
        values = torch.randn(10, 1)
        returns = torch.randn(10, 1)
        loss = ppo.compute_critic_loss(values, returns, values)
        expected = (returns - values).pow(2).mean()
        assert torch.allclose(loss, expected)


class TestDistributionalValueEstimate:
    """Tests for the quantile-to-scalar reduction in compute_value_estimate."""

    def test_value_estimate_is_mean_of_quantiles(self) -> None:
        """compute_value_estimate returns the mean over the quantile dimension."""
        ppo, obs, _ = _build_distributional_ppo()
        with torch.no_grad():
            expected = ppo.critic(obs).mean(dim=-1, keepdim=True)
        value = ppo.compute_value_estimate(obs)
        assert value.shape == (NUM_ENVS, 1)
        assert torch.allclose(value, expected)

    def test_value_estimate_is_detached(self) -> None:
        """compute_value_estimate must not carry gradients into storage."""
        ppo, obs, _ = _build_distributional_ppo()
        value = ppo.compute_value_estimate(obs)
        assert not value.requires_grad


class TestDistributionalPPOSubclass:
    """Tests verifying DistributionalPPO end-to-end functionality."""

    def test_init_stores_quantile_config(self) -> None:
        """Constructor stores num_quantiles and quantile_huber_kappa."""
        ppo, _, _ = _build_distributional_ppo(num_quantiles=16, quantile_huber_kappa=0.7)
        assert ppo.num_quantiles == 16
        assert ppo.quantile_huber_kappa == pytest.approx(0.7)

    def test_distributional_value_estimate_and_gae(self) -> None:
        """Critic outputs [B, Q] quantiles; compute_value_estimate extracts scalar [B, 1] for GAE."""
        ppo, obs, storage = _build_distributional_ppo()

        actions = ppo.act(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        # Stored value must be reduced scalar [NUM_ENVS, 1]
        assert ppo.transition.values.shape == (NUM_ENVS, 1)

        _rollout(ppo, obs)

        # compute_returns must bootstrap with scalar value estimate
        assert storage.returns.shape == (NUM_STEPS, NUM_ENVS, 1)
        assert storage.advantages.shape == (NUM_STEPS, NUM_ENVS, 1)

    def test_distributional_ppo_update_runs(self) -> None:
        """update() runs end-to-end with Quantile Huber loss without overriding update()."""
        ppo, obs, _ = _build_distributional_ppo()
        ppo.train_mode()
        _rollout(ppo, obs)

        before = [p.clone() for p in ppo.critic.parameters()]
        losses = ppo.update()

        assert "value" in losses
        assert "surrogate" in losses
        assert losses["value"] >= 0.0
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())

        # Parameters should update from quantile huber loss gradients
        after = list(ppo.critic.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after)), "critic params should update"

    @pytest.mark.parametrize("num_quantiles", [1, 4, 32])
    def test_update_runs_with_various_quantile_counts(self, num_quantiles: int) -> None:
        """update() is robust to different numbers of quantiles."""
        ppo, obs, _ = _build_distributional_ppo(num_quantiles=num_quantiles)
        ppo.train_mode()
        _rollout(ppo, obs)
        losses = ppo.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())

    def test_update_runs_with_mixed_precision(self) -> None:
        """update() with mixed precision returns finite losses."""
        ppo, obs, _ = _build_distributional_ppo(use_mixed_precision=True)
        ppo.train_mode()
        _rollout(ppo, obs)
        losses = ppo.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())


class TestDistributionalPPOConstructAlgorithm:
    """Tests for DistributionalPPO.construct_algorithm config handling."""

    @staticmethod
    def _make_cfg(algorithm_cfg: dict, critic_cfg: dict | None = None) -> dict:
        """Build a minimal runner config for construct_algorithm."""
        return {
            "num_steps_per_env": NUM_STEPS,
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "multi_gpu": None,
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [32, 32],
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
            },
            "critic": critic_cfg if critic_cfg is not None else {"class_name": "MLPModel", "hidden_dims": [32, 32]},
            "algorithm": algorithm_cfg,
        }

    def test_construct_algorithm(self) -> None:
        """construct_algorithm sets critic output_dim from num_quantiles."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS)
        cfg = self._make_cfg({"class_name": "DistributionalPPO", "num_quantiles": 16})

        alg = DistributionalPPO.construct_algorithm(obs, env, cfg, device="cpu")  # type: ignore[arg-type]
        assert isinstance(alg, DistributionalPPO)
        assert alg.num_quantiles == 16
        assert alg.critic(obs).shape == (NUM_ENVS, 16)
        assert alg.compute_value_estimate(obs).shape == (NUM_ENVS, 1)

    def test_construct_algorithm_default_quantiles(self) -> None:
        """Without num_quantiles in config, the default of 8 is used for the critic output_dim."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS)
        cfg = self._make_cfg({"class_name": "DistributionalPPO"})

        alg = DistributionalPPO.construct_algorithm(obs, env, cfg, device="cpu")  # type: ignore[arg-type]
        assert alg.num_quantiles == 8
        assert alg.critic(obs).shape == (NUM_ENVS, 8)

    def test_construct_algorithm_respects_explicit_output_dim(self) -> None:
        """An explicit critic output_dim is not overwritten by num_quantiles."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS)
        cfg = self._make_cfg(
            {"class_name": "DistributionalPPO", "num_quantiles": 16},
            critic_cfg={"class_name": "MLPModel", "hidden_dims": [32, 32], "output_dim": 32},
        )

        alg = DistributionalPPO.construct_algorithm(obs, env, cfg, device="cpu")  # type: ignore[arg-type]
        assert alg.critic(obs).shape == (NUM_ENVS, 32)
