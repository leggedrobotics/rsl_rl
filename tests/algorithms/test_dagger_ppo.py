# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the DaggerPPO algorithm."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.dagger_ppo import DaggerPPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = NUM_ACTIONS, **kwargs: object) -> MLPModel:
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _make_teacher(obs: TensorDict, obs_groups: dict, num_actions: int = NUM_ACTIONS, **kwargs: object) -> MLPModel:
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "tanh",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "teacher", num_actions, **defaults)


def _build_dagger_ppo(**overrides: object) -> tuple[DaggerPPO, TensorDict]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"], "teacher": ["policy"]}
    actor = _make_actor(obs, obs_groups)
    critic = _make_critic(obs, obs_groups)
    teacher = _make_teacher(obs, obs_groups)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        value_loss_coef=0.0,
        entropy_coef=0.0,
        dagger_loss_coef=1.0,
    )
    defaults.update(overrides)

    alg = DaggerPPO(actor, critic, teacher, storage, **defaults)
    return alg, obs


def _fill_rollout_storage(alg: DaggerPPO, obs: TensorDict) -> None:
    for _ in range(NUM_STEPS):
        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.hidden_states = (None, None)
        transition.actions = alg.actor(obs, stochastic_output=True).detach()
        transition.values = alg.critic(obs).detach()
        transition.actions_log_prob = alg.actor.get_output_log_prob(transition.actions).detach()
        transition.distribution_params = tuple(p.detach() for p in alg.actor.output_distribution_params)
        transition.rewards = torch.zeros(NUM_ENVS)
        transition.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(transition)

    alg.storage.advantages.zero_()
    alg.storage.returns.copy_(alg.storage.values)


class TestDaggerPPOLoss:
    """Tests for PPO updates with an auxiliary DAgger loss."""

    def test_update_reports_dagger_loss_and_keeps_teacher_frozen(self) -> None:
        alg, obs = _build_dagger_ppo()
        alg.train_mode()
        _fill_rollout_storage(alg, obs)

        actor_before = {name: param.clone() for name, param in alg.actor.named_parameters()}
        teacher_before = {name: param.clone() for name, param in alg.teacher.named_parameters()}

        loss_dict = alg.update()

        assert "dagger" in loss_dict
        assert loss_dict["dagger"] > 0.0
        assert any(not torch.equal(param, actor_before[name]) for name, param in alg.actor.named_parameters())
        for name, param in alg.teacher.named_parameters():
            assert torch.equal(param, teacher_before[name]), f"Teacher parameter {name} changed during update"


class TestDaggerPPOLoading:
    """Tests for teacher loading behavior."""

    def test_default_load_from_ppo_checkpoint_only_loads_teacher(self) -> None:
        alg, _obs = _build_dagger_ppo()

        teacher_before = {name: param.clone() for name, param in alg.teacher.named_parameters()}
        actor_before = {name: param.clone() for name, param in alg.actor.named_parameters()}

        teacher_source_obs = make_obs(NUM_ENVS, OBS_DIM)
        teacher_source_groups = {"actor": ["policy"], "critic": ["policy"]}
        source_actor = _make_actor(teacher_source_obs, teacher_source_groups)
        checkpoint = {"actor_state_dict": source_actor.state_dict()}

        load_iteration = alg.load(checkpoint, load_cfg=None, strict=True)

        assert load_iteration is False
        assert alg.teacher_loaded is True
        assert any(not torch.equal(param, teacher_before[name]) for name, param in alg.teacher.named_parameters())
        for name, param in alg.actor.named_parameters():
            assert torch.equal(param, actor_before[name]), f"Actor parameter {name} changed when loading teacher only"