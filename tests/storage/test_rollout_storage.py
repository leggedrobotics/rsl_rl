# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the storage."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 6
NUM_STEPS = 8
OBS_DIM = 4
NUM_ACTIONS = 3


def _make_storage_and_obs() -> tuple:
    """Create a storage and observations for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    return storage, obs


def _fill_with_identifiable_data(storage: RolloutStorage, obs: TensorDict) -> dict:
    """Fill storage with data where each transition has a unique identifier.

    Returns a dict mapping step to the stored values for verification.
    """
    stored = {}
    for step in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        # Use step-based values so each transition is identifiable
        t.actions = torch.full((NUM_ENVS, NUM_ACTIONS), float(step))
        t.values = torch.full((NUM_ENVS, 1), float(step) * 10)
        t.actions_log_prob = torch.full((NUM_ENVS,), float(step) * 0.1)
        t.distribution_params = (
            torch.full((NUM_ENVS, NUM_ACTIONS), float(step)),
            torch.full((NUM_ENVS, NUM_ACTIONS), 1.0),
        )
        t.rewards = torch.full((NUM_ENVS,), float(step) * 100)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)
        stored[step] = {
            "actions": t.actions.clone(),
            "values": t.values.clone(),
            "rewards": t.rewards.clone(),
        }

    # Set returns and advantages for mini-batch testing
    storage.returns = torch.randn_like(storage.returns)
    storage.advantages = torch.randn_like(storage.advantages)

    return stored


class TestMiniBatchGenerator:
    """Tests for ``mini_batch_generator`` data completeness."""

    def test_all_transitions_visited_in_one_epoch(self) -> None:
        """Every transition should appear exactly once in a single epoch."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        num_mini_batches = 4
        batch_size = NUM_ENVS * NUM_STEPS
        mini_batch_size = batch_size // num_mini_batches

        all_actions = []
        batch_count = 0
        for batch in storage.mini_batch_generator(num_mini_batches, num_epochs=1):
            all_actions.append(batch.actions)
            batch_count += 1

        assert batch_count == num_mini_batches

        collected = torch.cat(all_actions, dim=0)
        expected_count = num_mini_batches * mini_batch_size
        assert collected.shape[0] == expected_count

        # Check all unique action values from the stored transitions appear
        flat_stored = storage.actions.flatten(0, 1)
        for val in flat_stored[:expected_count]:
            assert torch.any(torch.all(collected == val.unsqueeze(0), dim=-1))

    def test_epoch_count_is_respected(self) -> None:
        """The generator should yield exactly num_epochs * num_mini_batches batches."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        num_mini_batches = 3
        num_epochs = 5
        batch_count = sum(1 for _ in storage.mini_batch_generator(num_mini_batches, num_epochs))

        assert batch_count == num_mini_batches * num_epochs

    def test_batch_fields_are_consistent(self) -> None:
        """All fields within a batch should index the same transitions."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        for batch in storage.mini_batch_generator(2, num_epochs=1):
            # actions and values should be from the same transitions
            # We stored actions = step, values = step * 10
            # So values / 10 should approximately equal actions (for each step)
            action_ids = batch.actions[:, 0]  # All actions in a step are the same value
            value_ids = batch.values[:, 0] / 10.0
            assert torch.allclose(action_ids, value_ids, atol=1e-5), (
                "Actions and values should index the same transitions"
            )


class TestRecurrentMiniBatchGenerator:
    """Tests for ``recurrent_mini_batch_generator`` — trajectory counting, env/trajectory alignment."""

    def test_trajectory_count_matches_env_slice(self) -> None:
        """The number of trajectories in each batch should match dones in the corresponding env slice."""
        storage, obs = _make_storage_and_obs()

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == 3:
                dones[0] = 1.0
            t.dones = dones
            storage.add_transition(t)

        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        mini_batch_size = NUM_ENVS // 2
        batches = list(storage.recurrent_mini_batch_generator(2, num_epochs=1))
        # Batch 0 covers envs [0..2]: env 0 has 2 trajectories, envs 1-2 have 1 each → 4 total
        assert batches[0].observations["policy"].shape[1] == mini_batch_size + 1
        # Batch 1 covers envs [3..5]: no dones → 3 trajectories (one each)
        assert batches[1].observations["policy"].shape[1] == mini_batch_size

    def test_obs_and_actions_refer_to_same_env_slice(self) -> None:
        """Trajectory-indexed obs and env-indexed actions must refer to the same environments, even with dones."""
        storage, _obs = _make_storage_and_obs()

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = TensorDict(
                {"policy": torch.arange(NUM_ENVS).float().unsqueeze(1).expand(-1, OBS_DIM)},
                batch_size=[NUM_ENVS],
            )
            t.hidden_states = (None, None)
            t.actions = torch.arange(NUM_ENVS).float().unsqueeze(1).expand(-1, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == 3:
                dones[0] = 1.0
            t.dones = dones
            storage.add_transition(t)

        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        for batch in storage.recurrent_mini_batch_generator(2, num_epochs=1):
            obs_vals = batch.observations["policy"]  # [time, num_trajs, obs_dim]
            masks = batch.masks  # [time, num_trajs]
            obs_env_ids = set(obs_vals[masks][:, 0].unique().long().tolist())

            action_env_ids = set(batch.actions[0, :, 0].unique().long().tolist())

            assert obs_env_ids == action_env_ids, f"Obs env ids {obs_env_ids} != action env ids {action_env_ids}"

    def test_epoch_count_is_respected(self) -> None:
        """Generator should yield exactly num_mini_batches * num_epochs batches."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        num_mini_batches = 3
        num_epochs = 5
        batch_count = sum(1 for _ in storage.recurrent_mini_batch_generator(num_mini_batches, num_epochs))
        assert batch_count == num_mini_batches * num_epochs

    def test_hidden_states_align_with_trajectory_starts(self) -> None:
        """Hidden states in each batch should correspond to trajectory start points, including after dones."""
        storage, obs = _make_storage_and_obs()
        rnn_hidden_dim = 8
        done_step = 3

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            h_actor = torch.full((1, NUM_ENVS, rnn_hidden_dim), float(step))
            t.hidden_states = (h_actor, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == done_step:
                dones[0] = 1.0
            t.dones = dones
            storage.add_transition(t)

        storage.returns = torch.randn_like(storage.returns)
        storage.advantages = torch.randn_like(storage.advantages)

        batches = list(storage.recurrent_mini_batch_generator(2, num_epochs=1))

        # Batch 0 covers envs [0,1,2]. Env 0 has 2 trajectories (start at step 0, restart at step 4).
        # Trajectory order from boolean indexing: env0-start0, env0-start4, env1-start0, env2-start0
        h0 = batches[0].hidden_states[0]
        assert h0 is not None
        assert h0.shape[1] == 4, f"Expected 4 trajectories, got {h0.shape[1]}"
        assert h0[0, 0, 0].item() == 0.0, "Env 0 first trajectory should start with step-0 hidden state"
        assert h0[0, 1, 0].item() == float(done_step + 1), (
            f"Env 0 second trajectory should start with step-{done_step + 1} hidden state"
        )

        # Batch 1 covers envs [3,4,5]. No dones → all hidden states from step 0.
        h1 = batches[1].hidden_states[0]
        assert h1 is not None
        assert torch.allclose(h1, torch.zeros_like(h1)), "Envs without dones should all have step-0 hidden states"


class TestDistillationStorage:
    """Tests for distillation-mode storage."""

    def test_generator_yields_per_timestep_batches(self) -> None:
        """Distillation generator should yield one batch per timestep."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

        for step in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.privileged_actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)

        batch_count = sum(1 for _ in storage.generator())
        assert batch_count == NUM_STEPS

    def test_rl_storage_rejects_distillation_generator(self) -> None:
        """An RL-mode storage should not allow the distillation generator."""
        storage, _obs = _make_storage_and_obs()
        with pytest.raises(ValueError):
            next(storage.generator())


class TestStorageOverflow:
    """Tests for storage capacity enforcement."""

    def test_overflow_raises_error(self) -> None:
        """Adding more transitions than capacity should raise OverflowError."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        with pytest.raises(OverflowError):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            t.values = torch.randn(NUM_ENVS, 1)
            t.actions_log_prob = torch.randn(NUM_ENVS)
            t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            storage.add_transition(t)


class TestStorageClear:
    """Tests for the clear() method."""

    def test_clear_allows_new_transitions(self) -> None:
        """After clear(), storage should accept new transitions starting from index 0."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        storage.clear()
        assert storage.step == 0

        # Should not raise
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
        t.values = torch.randn(NUM_ENVS, 1)
        t.actions_log_prob = torch.randn(NUM_ENVS)
        t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)
        assert storage.step == 1

    def test_clear_data_is_overwritten(self) -> None:
        """New data written after clear() should replace old data at the same index."""
        storage, obs = _make_storage_and_obs()
        _fill_with_identifiable_data(storage, obs)

        old_actions_step0 = storage.actions[0].clone()
        storage.clear()

        new_actions = torch.full((NUM_ENVS, NUM_ACTIONS), 999.0)
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = new_actions
        t.values = torch.randn(NUM_ENVS, 1)
        t.actions_log_prob = torch.randn(NUM_ENVS)
        t.distribution_params = (torch.randn(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        storage.add_transition(t)

        assert torch.allclose(storage.actions[0], new_actions)
        assert not torch.allclose(storage.actions[0], old_actions_step0)
