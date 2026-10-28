# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code ported from FlashSAC (flash_rl):
#   Copyright (c) 2026 Holiday Robotics
#   SPDX-License-Identifier: MIT
# ``RewardNormalizer`` / ``RunningMeanStd`` are adapted from
# ``flash_rl/agents/utils/reward_normalization.py``. The ``@torch.compile``
# decorators of the original are dropped (eager-first integration).

"""Reward normalization based on the running variance of discounted returns."""

from __future__ import annotations

import torch


def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    running_return: torch.Tensor,
    running_return_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the running discounted return and its running max magnitude."""
    done = torch.logical_or(terminated.bool(), truncated.bool()).float()
    new_return = gamma * (1.0 - done) * running_return + reward
    new_return_max = torch.maximum(running_return_max, torch.max(torch.abs(new_return)))
    return new_return, new_return_max


def _scale_reward(
    rewards: torch.Tensor,
    return_var: torch.Tensor,
    return_max: torch.Tensor,
    g_max: float,
    eps: float,
) -> torch.Tensor:
    """Scale rewards by the larger of the return std and ``return_max / g_max``."""
    var_denominator = torch.sqrt(return_var + eps)
    min_required_denominator = return_max / g_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return rewards / denominator


def _update_mean_var_count(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update running mean/var/count using Welford's parallel algorithm."""
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    m2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = m2 / total_count
    return new_mean, new_var, total_count


class RunningMeanStd:
    """Tracks the running mean, variance, and count of a stream of values."""

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize running statistics on ``device``."""
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        """Update the statistics from a batch of samples."""
        self.mean, self.var, self.count = _update_mean_var_count(
            samples=x,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )


class RewardNormalizer:
    """Normalize rewards by the running std of the discounted return.

    Scaling the return to roughly unit variance is a common variance-reduction
    technique; here it also keeps returns within the distributional critic's
    support ``[-g_max, g_max]``.
    """

    def __init__(
        self,
        gamma: float,
        g_max: float,
        num_envs: int,
        device: torch.device,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize the discounted-return trackers for ``num_envs`` environments."""
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device
        self._running_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._running_return_max = torch.zeros(1, dtype=torch.float32, device=device)
        self._return_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)

    def update_reward_stats(self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        """Update the running discounted-return statistics from one env step."""
        self._running_return, self._running_return_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            running_return=self._running_return,
            running_return_max=self._running_return_max,
            gamma=self.gamma,
        )
        self._return_rms.update(self._running_return.reshape(-1, 1))

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Return rewards scaled by the running return std (floored by ``max/g_max``)."""
        return _scale_reward(
            rewards=rewards,
            return_var=self._return_rms.var,
            return_max=self._running_return_max,
            g_max=self.g_max,
            eps=self.epsilon,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the normalizer state for checkpointing."""
        return {
            "running_return": self._running_return,
            "running_return_max": self._running_return_max,
            "return_rms_mean": self._return_rms.mean,
            "return_rms_var": self._return_rms.var,
            "return_rms_count": self._return_rms.count,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the normalizer state from :meth:`state_dict`."""
        self._running_return = state["running_return"].to(self.device)
        self._running_return_max = state["running_return_max"].to(self.device)
        self._return_rms.mean = state["return_rms_mean"].to(self.device)
        self._return_rms.var = state["return_rms_var"].to(self.device)
        self._return_rms.count = state["return_rms_count"].to(self.device)
