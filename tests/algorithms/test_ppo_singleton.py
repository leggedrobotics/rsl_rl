# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Advantage normalization at the smallest valid PPO rollout and batch sizes."""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.storage import RolloutStorage


def _make_rollout(model_type: str, num_envs: int, num_batches: int, per_batch: bool) -> tuple[PPO, TensorDict]:
    """Collect one real PPO step with known advantages and nonzero policy gradients."""
    torch.manual_seed(7)
    obs = TensorDict({"policy": torch.ones(num_envs, 2)}, batch_size=[num_envs])
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    model_class = MLPModel if model_type == "mlp" else RNNModel
    model_kwargs = {"hidden_dims": [4]}
    if model_type != "mlp":
        model_kwargs.update(rnn_type=model_type, rnn_hidden_dim=4)
    actor = model_class(
        obs,
        obs_groups,
        "actor",
        1,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        **model_kwargs,
    )
    critic = model_class(obs, obs_groups, "critic", 1, **model_kwargs)
    with torch.no_grad():
        for model in (actor, critic):
            for parameter in model.mlp.parameters():
                parameter.zero_()
            if model.is_recurrent:
                for parameter in model.rnn.parameters():
                    parameter.zero_()
                # Model a subsequent rollout, with saved zero hidden states for recurrent updates.
                model(obs, stochastic_output=model is actor)
                model.reset(torch.ones(num_envs))

    storage = RolloutStorage("rl", num_envs, 1, obs, [1])
    ppo = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=num_batches,
        gamma=0.0,
        lam=0.0,
        entropy_coef=0.0,
        schedule="fixed",
        normalize_advantage_per_mini_batch=per_batch,
    )
    with torch.no_grad():
        ppo.act(obs)
        # Select actions with a nonzero score under the actual rollout distribution.
        ppo.transition.actions = torch.arange(1, num_envs + 1, dtype=torch.float).reshape(num_envs, 1)
        ppo.transition.actions_log_prob = actor.get_output_log_prob(ppo.transition.actions)
        ppo.process_env_step(obs, torch.arange(1, num_envs + 1, dtype=torch.float), torch.zeros(num_envs), {})
        ppo.compute_returns(obs)
    return ppo, obs


@pytest.mark.parametrize("model_type", ["mlp", "gru", "lstm"])
@pytest.mark.parametrize(
    ("num_envs", "num_batches", "per_batch"),
    [
        pytest.param(1, 1, False, id="singleton-rollout"),
        pytest.param(1, 1, True, id="singleton-rollout-per-batch"),
        pytest.param(4, 4, True, id="singleton-minibatches"),
        pytest.param(4, 2, False, id="global-normalization-control"),
        pytest.param(4, 2, True, id="per-batch-normalization-control"),
    ],
)
def test_advantage_normalization_preserves_learning(
    model_type: str, num_envs: int, num_batches: int, per_batch: bool
) -> None:
    """Singletons keep their advantage; larger batches retain sample-std normalization."""
    ppo, _obs = _make_rollout(model_type, num_envs, num_batches, per_batch)
    raw_advantages = torch.arange(1, num_envs + 1, dtype=torch.float).reshape(1, num_envs, 1)
    torch.testing.assert_close(ppo.storage.returns, raw_advantages)
    expected = raw_advantages
    if not per_batch and expected.numel() > 1:
        expected = (expected - expected.mean()) / (expected.std() + 1e-8)
    torch.testing.assert_close(ppo.storage.advantages, expected)
    assert torch.isfinite(ppo.storage.advantages).all()

    generator_name = "recurrent_mini_batch_generator" if model_type != "mlp" else "mini_batch_generator"
    original_generator = getattr(ppo.storage, generator_name)
    seen_sizes = []

    def observe_batches(*args: object) -> Generator[RolloutStorage.Batch, None, None]:
        for batch in original_generator(*args):
            before = batch.advantages.clone()
            seen_sizes.append(before.numel())
            yield batch
            expected_batch = before
            if per_batch and before.numel() > 1:
                expected_batch = (before - before.mean()) / (before.std() + 1e-8)
            torch.testing.assert_close(batch.advantages, expected_batch)

    setattr(ppo.storage, generator_name, observe_batches)
    parameters_before = [
        [parameter.detach().clone() for parameter in model.parameters()] for model in (ppo.actor, ppo.critic)
    ]
    losses = ppo.update()
    assert seen_sizes == [num_envs // num_batches] * num_batches
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses.values())
    for model, old_parameters in zip((ppo.actor, ppo.critic), parameters_before):
        parameters = list(model.parameters())
        assert all(torch.isfinite(parameter).all() for parameter in parameters)
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        assert any(not torch.equal(old, new) for old, new in zip(old_parameters, parameters))
    for state in ppo.optimizer.state.values():
        assert all(torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor))
