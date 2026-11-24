# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for chronological lazy replay."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import SequenceReplayBuffer


def test_lazy_replay_ring_and_episode_validity() -> None:
    """Windows retain chronology, exclude resets, and clamp invalid positions to their last valid transition."""
    replay = SequenceReplayBuffer(2, 8)
    assert replay.data is None
    for t in range(6):
        replay.add(TensorDict({"value": torch.full((2, 1), t), "dones": torch.tensor([t == 3, False])}, batch_size=[2]))
    torch.manual_seed(0)
    index = replay.sample(100, length=3)
    sample = replay.gather(["value", "dones"], index.ids)
    for row in range(100):
        values = sample["value"][row, :, 0]
        anchor = int(values[0])
        env = int(index.ids[row, 0] % 2)
        for col in range(3):
            age = anchor + col
            expected = 2 <= age <= 5
            if env == 0 and anchor <= 3 < age:
                expected = False
            assert bool(index.valid[row, col]) == expected
            if expected:
                assert values[col] == age
        last = int(index.valid[row].sum()) - 1
        assert (index.ids[row, last:] == index.ids[row, last]).all()
    assert replay.num_steps == 4 and replay.size == 8


def test_collection_allocation_and_clear() -> None:
    """Inference-mode insertion leaves replay writable during later no-grad collection and reuse."""
    replay = SequenceReplayBuffer(2, 8)
    with torch.inference_mode():
        replay.add(TensorDict({"value": torch.ones(2, 3), "dones": torch.zeros(2, dtype=torch.bool)}, batch_size=[2]))
    assert all(not value.is_inference() for value in replay.data.values())
    with torch.no_grad():
        step = TensorDict({"value": torch.full((2, 3), 7.0), "dones": torch.zeros(2, dtype=torch.bool)}, batch_size=[2])
        replay.add(step)
        replay.clear()
        with pytest.raises(RuntimeError, match="empty replay"):
            replay.sample(1)
        replay.add(step)
    sample = replay.gather(["value"], replay.sample(8, length=3).ids)
    assert torch.equal(sample["value"], torch.full((8, 3, 3), 7.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The replay memory budget applies to CUDA.")
def test_lazy_memory_budget_checks_minimum_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional cap uses the actual transition schema and rejects a window that no longer fits."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1024, 4096))
    step = TensorDict({"value": torch.zeros(2, 1), "dones": torch.zeros(2, dtype=torch.bool)}, batch_size=[2])
    replay = SequenceReplayBuffer(2, 1000, "cuda", min_steps_per_env=3, max_memory_fraction=0.5)
    assert replay.data is None and replay.num_slots == 500
    with pytest.warns(UserWarning, match="Reducing replay"):
        replay.add(step)
    assert replay.num_slots == 51
    undersized = SequenceReplayBuffer(2, 1000, "cuda", min_steps_per_env=52, max_memory_fraction=0.5)
    with pytest.warns(UserWarning, match="Reducing replay"), pytest.raises(ValueError, match="52 steps"):
        undersized.add(step)


def test_invalid_transition_preserves_existing_slot() -> None:
    """Reject a changed transition schema before overwriting a full replay slot."""
    replay = SequenceReplayBuffer(1, 1)
    step = TensorDict({"dones": torch.tensor([False]), "state": torch.zeros(1, 1)}, [1])
    replay.add(step)
    invalid = TensorDict({"dones": torch.tensor([True]), "state": torch.ones(1, 2)}, [1])

    with pytest.raises(ValueError, match="shape or dtype changed"):
        replay.add(invalid)
    assert not replay.data["dones"].any()
    assert not replay.data["state"].any()


def test_history_repeats_earliest_episode_or_stored_frame() -> None:
    """SuperMimic histories repeat valid starts after overwrite and episode boundaries."""
    replay = SequenceReplayBuffer(1, 4)
    for step in range(7):
        replay.add(
            TensorDict({"value": torch.tensor([[float(step)]]), "dones": torch.tensor([step == 4])}, batch_size=[1])
        )
    index = replay.sample(100, history=3)
    values = replay.gather(("value",), index.ids)["value"][..., 0]
    for row in range(100):
        anchor = int(values[row, -1])
        earliest = max(3, 5 if anchor >= 5 else 3)
        expected = torch.arange(anchor - 3, anchor + 1).clamp(min=earliest).float()
        torch.testing.assert_close(values[row], expected)
        assert index.valid[row].sum() == min(4, anchor - earliest + 1)


def test_complete_windows_balance_environments_and_exclude_boundaries() -> None:
    """Unequal valid-window counts must not oversample an environment or cross a reset or edit."""
    replay = SequenceReplayBuffer(3, 18)
    for step in range(10):
        replay.add(
            TensorDict(
                {
                    "value": torch.full((3, 1), step),
                    "dones": torch.tensor([True, step in (5, 8), False]),
                    "window_break": torch.tensor([False, False, step == 4]),
                },
                [3],
            )
        )
    torch.manual_seed(42)
    index = replay.sample_sequences(6000, length=3)
    values = replay.gather(["value"], index.ids)["value"][..., 0]
    envs = index.ids[:, 0] % 3
    assert index.valid.all() and (envs != 0).all()
    assert ((envs == 1).float().mean() - 0.5).abs() < 0.04
    assert (values[envs == 1] == torch.tensor([6, 7, 8])).all()
    assert set(values[envs == 2, 0].tolist()) == {5, 6, 7}
    assert (values[:, 1:] - values[:, :-1] == 1).all()
