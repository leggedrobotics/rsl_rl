# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for lazy dense SAC replay."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import ReplayBuffer


def _step(value: float, done: bool = False) -> TensorDict:
    return TensorDict(
        {
            "observations": {"policy": torch.full((2, 1), value)},
            "actions": torch.arange(2, dtype=torch.float).unsqueeze(1),
            "rewards": torch.full((2,), value),
            "next_observations": {"policy": torch.full((2, 1), value + 100)},
            "dones": torch.tensor([done, False]),
            "terminated": torch.tensor([done, False]),
            "extra": {"value": torch.full((2,), value)},
        },
        batch_size=[2],
    )


def test_lazy_replay_ring_and_episode_validity() -> None:
    """Windows retain chronology, exclude resets, and clamp invalid positions to their last valid transition."""
    replay = ReplayBuffer(2, 8)
    assert replay.data is None
    for t in range(6):
        replay.add(_step(float(t), done=t == 3))
    torch.manual_seed(0)
    sample = replay.sample(100, length=3)
    for row in range(100):
        values = sample.window["extra", "value"][row]
        anchor = int(values[0])
        env = int(sample.actions[row, 0])
        for col in range(3):
            age = anchor + col
            expected = 2 <= age <= 5
            if env == 0 and anchor <= 3 < age:
                expected = False
            assert bool(sample.valid[row, col]) == expected
            if expected:
                assert values[col] == age
        last = int(sample.valid[row].sum()) - 1
        assert (values[last:] == values[last]).all()
        assert sample.observations["policy"][row, 0] == anchor
        assert sample.next_observations["policy"][row, 0] == values[last] + 100
        assert sample.terminated[row] == (env == 0 and values[last] == 3)
    assert replay.num_steps == 4 and replay.size == 8


def test_collection_allocation_and_clear() -> None:
    """Inference-mode insertion leaves replay writable during later no-grad collection and reuse."""
    replay = ReplayBuffer(2, 8)
    with torch.inference_mode():
        replay.add(_step(1.0))
    assert all(not value.is_inference() for value in replay.data.values(True, True))
    with torch.no_grad():
        step = _step(7.0)
        replay.add(step)
        replay.clear()
        with pytest.raises(RuntimeError, match="empty replay"):
            replay.sample(1)
        replay.add(step)
    sample = replay.sample(8, length=3)
    assert torch.equal(sample.window["rewards"], torch.full((8, 3), 7.0))
    assert torch.equal(sample.next_observations["policy"], torch.full((8, 1), 107.0))
    assert sample.valid.tolist() == [[True, False, False]] * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The replay memory budget applies to CUDA.")
def test_lazy_memory_budget_checks_minimum_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional cap uses the actual transition schema and rejects a window that no longer fits."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (10240, 40960))
    step = _step(0.0)
    replay = ReplayBuffer(2, 1000, "cuda", min_steps=3, max_memory_fraction=0.5)
    assert replay.data is None and replay.num_slots == 500
    with pytest.warns(UserWarning, match="Reducing replay"):
        replay.add(step)
    assert replay.num_slots == 116
    undersized = ReplayBuffer(2, 1000, "cuda", min_steps=117, max_memory_fraction=0.5)
    with pytest.warns(UserWarning, match="Reducing replay"), pytest.raises(ValueError, match="117 steps"):
        undersized.add(step)
