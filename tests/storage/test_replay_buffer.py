# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for lazy dense SAC replay."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import ReplayBuffer


def _make_transition(value: float, done: bool = False) -> ReplayBuffer.Transition:
    """Create one replay transition per environment."""
    transition = ReplayBuffer.Transition()
    transition.observations = TensorDict({"policy": torch.full((2, 1), value)}, batch_size=[2])
    transition.actions = torch.arange(2, dtype=torch.float).unsqueeze(1)
    transition.rewards = torch.full((2,), value)
    transition.next_observations = TensorDict({"policy": torch.full((2, 1), value + 100)}, batch_size=[2])
    transition.dones = torch.tensor([done, False])
    transition.terminated = torch.tensor([done, False])
    return transition


def test_lazy_replay_ring_and_episode_validity() -> None:
    """Windows retain chronology, exclude resets, and clamp invalid positions to their last valid transition."""
    replay = ReplayBuffer(2, 8)
    assert replay.observations is None
    for step in range(6):
        replay.add_transition(_make_transition(float(step), done=step == 3))
    torch.manual_seed(0)
    sample = replay.sample(100, length=3)
    for row in range(100):
        values = sample.rewards[row]
        anchor = int(values[0])
        env_id = int(sample.actions[row, 0])
        for offset in range(3):
            age = anchor + offset
            expected = 2 <= age <= 5
            if env_id == 0 and anchor <= 3 < age:
                expected = False
            assert bool(sample.valid[row, offset]) == expected
            if expected:
                assert values[offset] == age
        last = int(sample.valid[row].sum()) - 1
        assert (values[last:] == values[last]).all()
        assert sample.observations["policy"][row, 0] == anchor
        assert sample.next_observations["policy"][row, 0] == values[last] + 100
        assert sample.terminated[row] == (env_id == 0 and values[last] == 3)
    assert replay.num_steps == 4 and replay.size == 8


def test_collection_allocation_and_clear() -> None:
    """Inference-mode insertion leaves replay writable during later no-grad collection and reuse."""
    replay = ReplayBuffer(2, 8)
    with torch.inference_mode():
        replay.add_transition(_make_transition(1.0))
    assert all(
        not value.is_inference()
        for value in (
            *replay.observations.values(True, True),
            replay.actions,
            replay.rewards,
            *replay.next_observations.values(True, True),
            replay.dones,
            replay.terminated,
        )
    )
    with torch.no_grad():
        step = _make_transition(7.0)
        replay.add_transition(step)
        replay.clear()
        with pytest.raises(RuntimeError, match="empty replay"):
            replay.sample(1)
        replay.add_transition(step)
    step.observations["policy"].fill_(9.0)
    step.rewards.fill_(9.0)
    step.clear()
    assert step.observations is None
    assert step.rewards is None
    sample = replay.sample(8, length=3)
    assert torch.equal(sample.observations["policy"], torch.full((8, 1), 7.0))
    assert torch.equal(sample.rewards, torch.full((8, 3), 7.0))
    assert torch.equal(sample.next_observations["policy"], torch.full((8, 1), 107.0))
    assert sample.valid.tolist() == [[True, False, False]] * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The replay memory budget applies to CUDA.")
def test_lazy_memory_budget_checks_minimum_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional cap uses the actual transition schema and rejects a window that no longer fits."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (10240, 40960))
    step = _make_transition(0.0)
    replay = ReplayBuffer(2, 1000, min_steps_per_env=3, max_memory_fraction=0.5, device="cuda")
    assert replay.observations is None and replay.num_slots == 500
    with pytest.warns(UserWarning, match="Reducing replay"):
        replay.add_transition(step)
    assert replay.num_slots == 142
    undersized = ReplayBuffer(2, 1000, min_steps_per_env=143, max_memory_fraction=0.5, device="cuda")
    with pytest.warns(UserWarning, match="Reducing replay"), pytest.raises(ValueError, match="143 steps"):
        undersized.add_transition(step)


def test_invalid_transition_preserves_existing_slot() -> None:
    """Reject a changed transition schema before overwriting a full replay slot."""
    replay = ReplayBuffer(1, 1)
    step = TensorDict({"dones": torch.tensor([False]), "state": torch.zeros(1, 1)}, [1])
    replay.add_transition(step)
    invalid = TensorDict({"dones": torch.tensor([True]), "state": torch.ones(1, 2)}, [1])

    with pytest.raises(ValueError, match="shape or dtype changed"):
        replay.add_transition(invalid)
    assert not replay.transitions["dones"].any()
    assert not replay.transitions["state"].any()


def test_history_repeats_earliest_episode_or_stored_frame() -> None:
    """SHAC histories repeat valid starts after overwrite and episode boundaries."""
    replay = ReplayBuffer(1, 4)
    for step in range(7):
        replay.add_transition(
            TensorDict({"value": torch.tensor([[float(step)]]), "dones": torch.tensor([step == 4])}, batch_size=[1])
        )
    batch, valid = replay.sample_sequence(100, ("value",), history_length=3)
    values = batch["value"][..., 0]
    for row in range(100):
        anchor = int(values[row, -1])
        earliest = max(3, 5 if anchor >= 5 else 3)
        expected = torch.arange(anchor - 3, anchor + 1).clamp(min=earliest).float()
        torch.testing.assert_close(values[row], expected)
        assert valid[row].sum() == min(4, anchor - earliest + 1)


def test_complete_windows_balance_environments_and_exclude_boundaries() -> None:
    """Unequal valid-window counts must not oversample an environment or cross a reset or edit."""
    replay = ReplayBuffer(3, 18)
    for step in range(10):
        replay.add_transition(
            TensorDict(
                {
                    "value": torch.full((3, 1), step),
                    "env": torch.arange(3),
                    "dones": torch.tensor([True, step in (5, 8), False]),
                    "window_break": torch.tensor([False, False, step == 4]),
                },
                [3],
            )
        )
    torch.manual_seed(42)
    counts = replay.compute_complete_window_counts(3)
    batch, valid = replay.sample_sequence(6000, ("value", "env"), length=3, complete_window_counts=counts)
    values = batch["value"][..., 0]
    envs = batch["env"][:, 0]
    assert valid.all() and (envs != 0).all()
    assert ((envs == 1).float().mean() - 0.5).abs() < 0.04
    assert (values[envs == 1] == torch.tensor([6, 7, 8])).all()
    assert set(values[envs == 2, 0].tolist()) == {5, 6, 7}
    assert (values[:, 1:] - values[:, :-1] == 1).all()
