# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FlashSAC neural-network building blocks."""

from __future__ import annotations

import math
import torch

from rsl_rl.modules.flashsac_layers import (
    EnsembleCategoricalValue,
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
    UnitLinear,
    safe_tanh_log_det_jacobian,
)

BATCH = 8
OBS_DIM = 12
ACT_DIM = 3


class TestActor:
    """Tests for the squashed-Gaussian actor network."""

    def test_forward_shapes_and_bounds(self) -> None:
        """forward() returns tanh-bounded actions and a per-sample log-prob."""
        actor = FlashSACActor(num_blocks=2, input_dim=OBS_DIM, hidden_dim=32, action_dim=ACT_DIM)
        x = torch.randn(BATCH, OBS_DIM)
        actions, info = actor(x, training=True)
        assert actions.shape == (BATCH, ACT_DIM)
        assert info["log_prob"].shape == (BATCH,)
        assert (actions.abs() <= 1.0 + 1e-6).all()

    def test_mean_std_positive(self) -> None:
        """get_mean_and_std() returns a strictly positive standard deviation."""
        actor = FlashSACActor(num_blocks=1, input_dim=OBS_DIM, hidden_dim=16, action_dim=ACT_DIM)
        mean, std = actor.get_mean_and_std(torch.randn(BATCH, OBS_DIM), training=False)
        assert mean.shape == (BATCH, ACT_DIM)
        assert (std > 0).all()


class TestCritic:
    """Tests for the ensembled distributional double critic."""

    def test_forward_shapes(self) -> None:
        """forward() returns (num_qs, B) values and (num_qs, B, num_bins) log-probs."""
        num_bins = 51
        critic = FlashSACDoubleCritic(
            num_blocks=2, input_dim=OBS_DIM + ACT_DIM, hidden_dim=32, num_bins=num_bins, min_v=-5.0, max_v=5.0
        )
        obs = torch.randn(BATCH, OBS_DIM)
        act = torch.rand(BATCH, ACT_DIM) * 2 - 1
        qs, info = critic(obs, act, training=True)
        assert qs.shape == (2, BATCH)
        assert info["log_prob"].shape == (2, BATCH, num_bins)

    def test_distribution_normalized(self) -> None:
        """The categorical value head produces per-bin probabilities that sum to 1."""
        num_bins = 41
        head = EnsembleCategoricalValue(num_ensemble=2, hidden_dim=16, num_bins=num_bins, min_v=-3.0, max_v=3.0)
        _, info = head(torch.randn(2, BATCH, 16), training=False)
        prob_sum = info["log_prob"].exp().sum(-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)


class TestWeightNormalization:
    """Tests for the weight-normalization contract."""

    def test_unit_linear_rows_normalized(self) -> None:
        """normalize_parameters() makes each output feature a unit vector."""
        layer = UnitLinear(6, 5)
        layer.normalize_parameters()
        norms = layer.w.weight.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestTanhJacobian:
    """Tests for the numerically stable tanh log-det-Jacobian."""

    def test_matches_closed_form(self) -> None:
        """safe_tanh_log_det_jacobian equals log(1 - tanh(x)^2)."""
        x = torch.linspace(-5.0, 5.0, 21)
        expected = torch.log(1 - torch.tanh(x) ** 2 + 1e-12)
        assert torch.allclose(safe_tanh_log_det_jacobian(x), expected, atol=1e-4)


class TestTemperature:
    """Tests for the learnable entropy temperature."""

    def test_initial_value(self) -> None:
        """forward() returns exp(log_temp) initialized to the requested value."""
        temp = FlashSACTemperature(initial_value=0.05)
        assert math.isclose(float(temp().item()), 0.05, rel_tol=1e-5)
