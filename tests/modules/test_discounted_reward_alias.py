# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discounted return history must own the observations it accumulates."""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.extensions.rnd import RandomNetworkDistillation
from rsl_rl.modules.normalization import EmpiricalDiscountedVariationNormalization


def test_mutating_a_reused_reward_buffer_does_not_change_discounted_history() -> None:
    """Keep independent history when an environment reuses its reward buffer."""
    normalizer = EmpiricalDiscountedVariationNormalization(shape=[], gamma=0.5).double()
    buffer = torch.tensor([2.0, 4.0], dtype=torch.float64)
    normalizer(buffer)
    buffer.fill_(100.0)
    torch.testing.assert_close(normalizer.disc_avg.avg, torch.tensor([2.0, 4.0], dtype=torch.float64))
    normalizer(torch.tensor([0.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(normalizer.disc_avg.avg, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_scaling_a_zero_variance_output_does_not_scale_discounted_history() -> None:
    """Accumulate raw returns when zero-variance outputs are scaled in place."""
    normalizer = EmpiricalDiscountedVariationNormalization(shape=[], gamma=0.5).double()
    result = normalizer(torch.tensor([2.0, 2.0], dtype=torch.float64))
    result.mul_(3.0)
    normalizer(torch.tensor([2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(normalizer.disc_avg.avg, torch.tensor([3.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(normalizer.emp_norm.mean, torch.tensor(2.5, dtype=torch.float64))
    torch.testing.assert_close(normalizer.emp_norm.std, torch.tensor(0.5, dtype=torch.float64))


@pytest.mark.parametrize("weight", [0.0, 0.5, 3.0])
def test_real_rnd_scaling_leaves_unweighted_return_moments_unchanged(weight: float) -> None:
    """Match independent unweighted return moments through actual RND rewards."""
    rnd = RandomNetworkDistillation(
        num_states=2,
        obs_groups={"rnd_state": ["policy"]},
        num_outputs=1,
        predictor_hidden_dims=[3],
        target_hidden_dims=[3],
        reward_normalization=True,
        weight=weight,
    ).double()
    with torch.no_grad():
        for parameter in rnd.parameters():
            parameter.zero_()
        rnd.target[-1].bias.fill_(2.0)
    observations = TensorDict({"policy": torch.zeros(4, 2, dtype=torch.float64)}, batch_size=[4])
    history = []
    average = 0.0
    for _ in range(4):
        average = 0.99 * average + 2.0
        history.append(average)
        mean = sum(history) / len(history)
        variance = sum((value - mean) ** 2 for value in history) / len(history)
        expected_reward = weight * (2.0 / math.sqrt(variance) if variance > 0 else 2.0)
        actual = rnd.get_intrinsic_reward(observations)
        torch.testing.assert_close(actual, torch.full((4,), expected_reward, dtype=torch.float64))
        torch.testing.assert_close(rnd.reward_normalizer.disc_avg.avg, torch.full((4,), average, dtype=torch.float64))
        torch.testing.assert_close(rnd.reward_normalizer.emp_norm.mean, torch.tensor(mean, dtype=torch.float64))
        torch.testing.assert_close(
            rnd.reward_normalizer.emp_norm.std, torch.tensor(math.sqrt(variance), dtype=torch.float64)
        )


def test_evaluation_does_not_change_a_captured_discounted_history() -> None:
    """Freeze the retained return history during evaluation."""
    normalizer = EmpiricalDiscountedVariationNormalization(shape=[], gamma=0.5).double()
    normalizer(torch.tensor([1.0, 3.0], dtype=torch.float64))
    before = normalizer.disc_avg.avg.clone()
    normalizer.eval()
    normalizer(torch.tensor([10.0, 20.0], dtype=torch.float64))
    torch.testing.assert_close(normalizer.disc_avg.avg, before)
