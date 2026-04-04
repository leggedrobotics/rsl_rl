# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the AMPPPO algorithm."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = NUM_ACTIONS, **kwargs: object) -> MLPModel:
    defaults: dict[str, Any] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    defaults: dict[str, Any] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _make_discriminator(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    defaults: dict[str, Any] = {"hidden_dims": [64, 64], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "amp", 1, **defaults)


def _build_amp_ppo(**overrides: object) -> tuple[AMPPPO, TensorDict]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"], "amp": ["policy"]}
    actor = _make_actor(obs, obs_groups)
    critic = _make_critic(obs, obs_groups)
    discriminator = _make_discriminator(obs, obs_groups)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults: dict[str, Any] = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        value_loss_coef=0.0,
        entropy_coef=0.0,
        amp_reward_coef=0.5,
        discriminator_loss_coef=1.0,
        expert_batch_size=4,
        expert_observations=torch.randn(256, OBS_DIM),
    )
    defaults.update(overrides)

    alg = AMPPPO(actor, critic, discriminator, storage, **defaults)
    return alg, obs


def _fill_rollout_storage(alg: AMPPPO, obs: TensorDict) -> None:
    for _ in range(NUM_STEPS):
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.hidden_states = (None, None)
        transition.actions = alg.actor(obs, stochastic_output=True).detach()
        assert transition.actions is not None
        transition.values = alg.critic(obs).detach()
        transition.actions_log_prob = alg.actor.get_output_log_prob(transition.actions).detach()
        transition.distribution_params = tuple(p.detach() for p in alg.actor.output_distribution_params)
        transition.rewards = torch.zeros(NUM_ENVS)
        transition.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(transition)

    alg.storage.advantages.zero_()
    alg.storage.returns.copy_(alg.storage.values)


class TestAMPPPO:
    """Tests for AMPPPO-specific behavior."""

    def test_process_env_step_adds_amp_reward(self) -> None:
        alg, obs = _build_amp_ppo(amp_reward_coef=1.0)

        alg.act(obs)
        rewards = torch.zeros(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(obs, rewards, dones, {"time_outs": torch.zeros(NUM_ENVS)})

        stored_rewards = alg.storage.rewards[0, :, 0]
        assert torch.any(stored_rewards > 0.0)

    def test_update_reports_amp_metrics_and_updates_discriminator(self) -> None:
        alg, obs = _build_amp_ppo()
        _fill_rollout_storage(alg, obs)

        disc_before = {name: param.clone() for name, param in alg.discriminator.named_parameters()}
        loss_dict = alg.update()

        assert "amp_discriminator" in loss_dict
        assert "amp_reward" in loss_dict
        assert loss_dict["amp_discriminator"] > 0.0
        changed = any(not torch.equal(param, disc_before[name]) for name, param in alg.discriminator.named_parameters())
        assert changed, "Discriminator parameters should change after update"

    def test_save_load_roundtrip_keeps_discriminator_weights(self) -> None:
        alg, _obs = _build_amp_ppo()

        saved = alg.save()
        clone, _ = _build_amp_ppo()
        clone.load(saved, load_cfg=None, strict=True)

        for key, value in alg.discriminator.state_dict().items():
            assert torch.equal(value, clone.discriminator.state_dict()[key]), f"Mismatch for key: {key}"

    def test_load_from_ppo_checkpoint_skips_discriminator(self) -> None:
        alg, _obs = _build_amp_ppo()

        ppo_only_checkpoint = {
            "actor_state_dict": alg.actor.state_dict(),
            "critic_state_dict": alg.critic.state_dict(),
            "optimizer_state_dict": alg.optimizer.state_dict(),
        }

        load_iter = alg.load(ppo_only_checkpoint, load_cfg=None, strict=True)
        assert load_iter is True
