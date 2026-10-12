# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ReplayStorage off-policy buffer."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import ReplayStorage

NUM_ENVS = 4
OBS_DIM = 6
ACT_DIM = 2


def _obs(num_envs: int = NUM_ENVS, fill: float | None = None) -> TensorDict:
    """Create an observation TensorDict (optionally constant-filled)."""
    policy = torch.full((num_envs, OBS_DIM), fill) if fill is not None else torch.randn(num_envs, OBS_DIM)
    return TensorDict({"policy": policy}, batch_size=[num_envs])


def _make_storage(capacity: int = 32, min_length: int = 8, n_step: int = 1, gamma: float = 0.99) -> ReplayStorage:
    """Create a ReplayStorage with a sample observation."""
    return ReplayStorage(
        num_envs=NUM_ENVS,
        obs=_obs(),
        actions_shape=[ACT_DIM],
        capacity=capacity,
        min_length=min_length,
        sample_batch_size=5,
        n_step=n_step,
        gamma=gamma,
        device="cpu",
    )


def _step(reward: float = 1.0, terminated: float = 0.0, truncated: float = 0.0) -> dict:
    """Build one vectorized transition kwargs dict."""
    return {
        "observations": _obs(),
        "actions": torch.rand(NUM_ENVS, ACT_DIM) * 2 - 1,
        "rewards": torch.full((NUM_ENVS,), reward),
        "terminated": torch.full((NUM_ENVS,), terminated),
        "truncated": torch.full((NUM_ENVS,), truncated),
        "next_observations": _obs(),
    }


class TestBasic:
    """Tests for basic add/sample behavior."""

    def test_can_sample_threshold(self) -> None:
        """can_sample() flips once min_length transitions are stored."""
        storage = _make_storage(min_length=8)  # 8 / 4 env = 2 adds
        assert not storage.can_sample()
        storage.add(**_step())
        assert not storage.can_sample()
        storage.add(**_step())
        assert storage.can_sample()
        assert len(storage) == 8

    def test_sample_shapes(self) -> None:
        """sample() returns correctly shaped, grouped tensors."""
        storage = _make_storage()
        for _ in range(4):
            storage.add(**_step())
        batch = storage.sample()
        assert batch.observations["policy"].shape == (5, OBS_DIM)
        assert batch.next_observations["policy"].shape == (5, OBS_DIM)
        assert batch.actions.shape == (5, ACT_DIM)
        assert batch.rewards.shape == (5,)
        assert batch.terminated.shape == (5,)
        assert batch.truncated.shape == (5,)

    def test_capacity_wraps(self) -> None:
        """The ring buffer wraps and caps at capacity."""
        storage = _make_storage(capacity=8)  # holds 2 adds
        for _ in range(5):
            storage.add(**_step())
        assert len(storage) == 8


class TestNStep:
    """Tests for n-step return accumulation."""

    def test_one_step_is_identity(self) -> None:
        """With n_step=1 the stored reward equals the raw reward."""
        storage = _make_storage(n_step=1, gamma=0.9)
        storage.add(**_step(reward=2.0))
        assert torch.allclose(storage.rewards[:NUM_ENVS], torch.full((NUM_ENVS,), 2.0))

    def test_multi_step_discounted_sum(self) -> None:
        """With n_step=3 and no dones the reward is r0 + g r1 + g^2 r2."""
        gamma = 0.5
        storage = _make_storage(capacity=64, n_step=3, gamma=gamma)
        storage.add(**_step(reward=1.0))  # no write yet (window filling)
        storage.add(**_step(reward=2.0))
        assert len(storage) == 0
        storage.add(**_step(reward=4.0))  # window full -> writes first transition
        expected = 1.0 + gamma * 2.0 + gamma**2 * 4.0
        assert torch.allclose(storage.rewards[:NUM_ENVS], torch.full((NUM_ENVS,), expected), atol=1e-6)

    def test_done_truncates_accumulation(self) -> None:
        """A termination inside the window stops the discounted accumulation."""
        gamma = 0.5
        storage = _make_storage(capacity=64, n_step=3, gamma=gamma)
        storage.add(**_step(reward=1.0, terminated=1.0))  # done at first step
        storage.add(**_step(reward=2.0))
        storage.add(**_step(reward=4.0))
        # done at step 0 => reward is just r0, terminated propagated.
        assert torch.allclose(storage.rewards[:NUM_ENVS], torch.full((NUM_ENVS,), 1.0), atol=1e-6)
        assert torch.allclose(storage.terminated[:NUM_ENVS], torch.ones(NUM_ENVS))


class TestValidation:
    """Tests for fail-loud construction."""

    def test_capacity_below_num_envs_raises(self) -> None:
        """Capacity smaller than num_envs raises ValueError."""
        with pytest.raises(ValueError):
            ReplayStorage(
                NUM_ENVS, _obs(), [ACT_DIM], capacity=2, min_length=1, sample_batch_size=1, n_step=1, gamma=0.99
            )

    def test_bad_n_step_raises(self) -> None:
        """n_step < 1 raises ValueError."""
        with pytest.raises(ValueError):
            ReplayStorage(
                NUM_ENVS, _obs(), [ACT_DIM], capacity=8, min_length=1, sample_batch_size=1, n_step=0, gamma=0.99
            )


class TestPersistence:
    """Tests for save/load round-trip."""

    def test_save_load_roundtrip(self) -> None:
        """Saved buffer contents reload identically."""
        storage = _make_storage()
        for _ in range(4):
            storage.add(**_step())
        n = len(storage)
        rewards_before = storage.rewards[:n].clone()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            storage.save(f.name)
            fresh = _make_storage()
            fresh.load(f.name)
        assert len(fresh) == n
        assert torch.allclose(fresh.rewards[:n], rewards_before)
