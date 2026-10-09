# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout coverage and alignment when mini-batch counts do not divide the data."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import RNN
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


def _identified_storage(num_envs: int, num_steps: int) -> RolloutStorage:
    """Record unique sample IDs, mixed episode boundaries and actor/critic states."""
    obs = TensorDict({"policy": torch.zeros(num_envs, 1)}, batch_size=[num_envs])
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [1])
    for step in range(num_steps):
        ids = torch.arange(num_envs, dtype=torch.float32) + step * num_envs
        transition = RolloutStorage.Transition()
        transition.observations = TensorDict({"policy": ids[:, None]}, batch_size=[num_envs])
        transition.actions = ids[:, None]
        transition.values = ids[:, None] * 10
        transition.actions_log_prob = ids * 20
        transition.distribution_params = (ids[:, None] * 30, torch.ones(num_envs, 1))
        transition.rewards = ids
        transition.dones = torch.zeros(num_envs)
        if step == 0:
            transition.dones[0] = 1
        if step == num_steps // 2:
            transition.dones[-1] = 1
        hidden = (torch.arange(num_envs, dtype=torch.float32) + 1000 * step)[None, :, None]
        hidden = hidden.expand(2, num_envs, 3).clone()
        transition.hidden_states = (hidden, (hidden + 20, hidden + 30))
        storage.add_transition(transition)
    storage.returns.copy_(storage.values * 10)
    storage.advantages.copy_(storage.values * 100)
    return storage


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("num_envs,num_steps,num_batches", [(5, 3, 2), (7, 4, 3), (6, 4, 3)])
@pytest.mark.parametrize("num_epochs", [1, 3])
def test_each_epoch_visits_every_sample_once(
    recurrent: bool, num_envs: int, num_steps: int, num_batches: int, num_epochs: int
) -> None:
    """No tail transition or environment can be excluded from every update."""
    storage = _identified_storage(num_envs, num_steps)
    generator = storage.recurrent_mini_batch_generator if recurrent else storage.mini_batch_generator
    batches = list(generator(num_batches, num_epochs))
    assert len(batches) == num_batches * num_epochs
    for epoch in range(num_epochs):
        epoch_batches = batches[epoch * num_batches : (epoch + 1) * num_batches]
        ids = torch.cat([batch.actions.flatten() for batch in epoch_batches]).long().sort().values
        assert torch.equal(ids, torch.arange(num_envs * num_steps))
        sizes = [batch.actions.shape[1 if recurrent else 0] for batch in epoch_batches]
        assert min(sizes) > 0
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("num_envs,num_steps,num_batches", [(5, 3, 2), (7, 4, 3)])
def test_remainder_batch_fields_stay_aligned(recurrent: bool, num_envs: int, num_steps: int, num_batches: int) -> None:
    """The new tail samples retain observations, targets and distribution parameters."""
    storage = _identified_storage(num_envs, num_steps)
    generator = storage.recurrent_mini_batch_generator if recurrent else storage.mini_batch_generator
    batches = list(generator(num_batches, 1))
    for batch in batches:
        observations = unpad_trajectories(batch.observations, batch.masks) if recurrent else batch.observations
        ids = batch.actions
        assert torch.equal(observations["policy"], ids)
        assert torch.equal(batch.values, ids * 10)
        assert torch.equal(batch.returns, ids * 100)
        assert torch.equal(batch.advantages, ids * 1000)
        assert torch.equal(batch.old_actions_log_prob, ids * 20)
        assert torch.equal(batch.old_distribution_params[0], ids * 30)
        assert torch.equal(batch.old_distribution_params[1], torch.ones_like(ids))
    all_ids = torch.cat([batch.actions.flatten() for batch in batches]).long().sort().values
    assert torch.equal(all_ids, torch.arange(num_envs * num_steps))


@pytest.mark.parametrize("num_envs,num_steps,num_batches", [(5, 3, 2), (7, 4, 3), (6, 4, 3)])
def test_recurrent_remainder_hidden_states_match_episode_starts(
    num_envs: int, num_steps: int, num_batches: int
) -> None:
    """Both GRU and LSTM states remain paired with their own environment/episode."""
    storage = _identified_storage(num_envs, num_steps)
    batches = list(storage.recurrent_mini_batch_generator(num_batches, 1))
    for batch in batches:
        environments = batch.actions[0, :, 0].long().tolist()
        starts = []
        for environment in environments:
            starts.append(environment)
            starts.extend(
                (step + 1) * 1000 + environment for step in range(num_steps - 1) if storage.dones[step, environment, 0]
            )
        expected = torch.tensor(starts, dtype=torch.float32)
        assert torch.equal(batch.hidden_states[0][0, :, 0], expected)
        assert torch.equal(batch.hidden_states[1][0][0, :, 0], expected + 20)
        assert torch.equal(batch.hidden_states[1][1][0, :, 0], expected + 30)
        assert batch.masks.sum().item() == num_steps * len(environments)
    assert sum(batch.actions.shape[1] for batch in batches) == num_envs


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
@pytest.mark.parametrize("num_envs,num_steps,num_batches", [(5, 3, 2), (7, 4, 3)])
def test_remainder_trajectories_work_with_native_rnn_backward(
    rnn_type: str, num_envs: int, num_steps: int, num_batches: int
) -> None:
    """Real recurrent consumers process the complete variable-width environment slices."""
    storage = _identified_storage(num_envs, num_steps)
    model = RNN(1, hidden_dim=3, num_layers=2, type=rnn_type)
    seen = []
    for batch in storage.recurrent_mini_batch_generator(num_batches, 1):
        hidden = batch.hidden_states[0 if rnn_type == "gru" else 1]
        output = model(batch.observations["policy"], batch.masks, hidden)
        assert output.shape[:2] == batch.actions.shape[:2]
        assert torch.isfinite(output).all()
        output.square().sum().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        )
        model.zero_grad()
        seen.extend(batch.actions.flatten().long().tolist())
    assert sorted(seen) == list(range(num_envs * num_steps))


def test_ppo_can_learn_a_signal_present_only_in_the_last_transition() -> None:
    """Dropping the fixed tail must not make its unique input completely untrainable."""
    num_envs, num_steps = 5, 3
    obs = TensorDict({"policy": torch.zeros(num_envs, 1)}, batch_size=[num_envs])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(obs, groups, "actor", 1, hidden_dims=[1], distribution_cfg={"class_name": "GaussianDistribution"})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[1])
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        output_layer = [module for module in critic.modules() if isinstance(module, torch.nn.Linear)][-1]
        output_layer.weight.fill_(1)
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [1])
    ppo = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2, schedule="fixed")
    for step in range(num_steps):
        current = torch.zeros(num_envs, 1)
        reward = torch.zeros(num_envs)
        if step == num_steps - 1:
            current[-1] = 1
            reward[-1] = 2
        current_obs = TensorDict({"policy": current}, batch_size=[num_envs])
        ppo.act(current_obs)
        ppo.process_env_step(obs, reward, torch.zeros(num_envs), {})
    ppo.compute_returns(obs)
    losses = ppo.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert next(critic.parameters()).abs().max().item() > 0


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("num_batches", [0, -1, 16])
def test_invalid_partition_counts_are_rejected(recurrent: bool, num_batches: int) -> None:
    """An impossible partition count must not yield empty optimization batches."""
    storage = _identified_storage(5, 3)
    generator = storage.recurrent_mini_batch_generator if recurrent else storage.mini_batch_generator
    with pytest.raises(ValueError, match="num_mini_batches"):
        list(generator(num_batches, 1))
