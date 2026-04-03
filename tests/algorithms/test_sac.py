# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the SAC algorithm."""

from __future__ import annotations

import copy
import tempfile
import torch

from rsl_rl.algorithms.sac import SAC
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 3


def _build_sac(**overrides: object) -> tuple[SAC, torch.Tensor]:
    obs = make_obs(NUM_ENVS, OBS_DIM)

    defaults = dict(
        actor_obs_groups=["policy"],
        critic_obs_groups=["policy"],
        actor_obs_dim=OBS_DIM,
        critic_obs_dim=OBS_DIM,
        action_dim=NUM_ACTIONS,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        batch_size=16,
        gradient_steps=2,
        learning_starts=16,
        replay_buffer_size=512,
    )
    defaults.update(overrides)
    alg = SAC(**defaults)
    return alg, obs


def _fill_buffer(alg: SAC, steps: int) -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    for _ in range(steps):
        actions = alg.act(obs)
        next_obs = make_obs(NUM_ENVS, OBS_DIM)
        rewards = torch.randn(NUM_ENVS)
        dones = torch.zeros(NUM_ENVS)
        alg.process_env_step(next_obs, rewards, dones, {})
        obs = next_obs
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)


class TestSACBasics:
    """Tests for construction and action sampling."""

    def test_act_returns_bounded_actions(self) -> None:
        alg, obs = _build_sac(learning_starts=0)
        actions = alg.act(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert torch.all(actions <= 1.0)
        assert torch.all(actions >= -1.0)

    def test_update_changes_parameters(self) -> None:
        alg, _obs = _build_sac()
        _fill_buffer(alg, steps=32)

        params_before = {k: v.clone() for k, v in alg.actor.state_dict().items()}
        loss_dict = alg.update()

        assert "actor" in loss_dict
        assert "critic" in loss_dict
        any_changed = any(not torch.equal(params_before[k], v) for k, v in alg.actor.state_dict().items())
        assert any_changed, "Actor parameters should change after update"


class TestSACCheckpointing:
    """Tests for save/load functionality."""

    def test_save_and_load_restores_actor(self) -> None:
        alg, _obs = _build_sac()
        _fill_buffer(alg, steps=32)
        alg.update()

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(alg.save(), f.name)
            saved_state = copy.deepcopy(alg.actor.state_dict())

            _fill_buffer(alg, steps=32)
            alg.update()
            assert not all(torch.equal(saved_state[k], v) for k, v in alg.actor.state_dict().items())

            loaded_dict = torch.load(f.name, weights_only=False, map_location="cpu")
            alg.load(loaded_dict, load_cfg=None, strict=True)
            for key, value in alg.actor.state_dict().items():
                assert torch.equal(saved_state[key], value)