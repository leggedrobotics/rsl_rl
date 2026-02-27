# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RNNModel."""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models import RNNModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_rnn_model(rnn_type: str = "gru", **kwargs: object) -> tuple[RNNModel, TensorDict]:
    """Create an RNNModel and matching observations."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"]}
    defaults = dict(
        hidden_dims=[32, 32],
        activation="elu",
        rnn_type=rnn_type,
        rnn_hidden_dim=16,
        rnn_num_layers=1,
    )
    defaults.update(kwargs)
    model = RNNModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)
    return model, obs


class TestHiddenStateReset:
    """Tests for hidden state reset behavior on done environments."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_reset_done_env_zeros_its_state(self, rnn_type: str) -> None:
        """reset(dones) should zero the hidden state of done envs, preserving others."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)

        # Run a step to initialize hidden state
        model(obs)
        assert model.get_hidden_state() is not None

        # Save state of env 1 before reset
        if rnn_type == "lstm":
            h_before, _c_before = model.get_hidden_state()
            h1_before = h_before[:, 1, :].clone()
        else:
            h_before = model.get_hidden_state()
            h1_before = h_before[:, 1, :].clone()

        # Mark env 0 as done, env 1 as not done
        dones = torch.zeros(NUM_ENVS)
        dones[0] = 1.0
        model.reset(dones)

        if rnn_type == "lstm":
            h_after, c_after = model.get_hidden_state()
            # Env 0 should be zeroed
            assert torch.all(h_after[:, 0, :] == 0)
            assert torch.all(c_after[:, 0, :] == 0)
            # Env 1 should be preserved
            assert torch.allclose(h_after[:, 1, :], h1_before)
        else:
            h_after = model.get_hidden_state()
            assert torch.all(h_after[:, 0, :] == 0)
            assert torch.allclose(h_after[:, 1, :], h1_before)

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_full_reset_clears_all_state(self, rnn_type: str) -> None:
        """reset() without arguments should clear the hidden state entirely."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)
        model(obs)
        assert model.get_hidden_state() is not None

        model.reset()
        assert model.get_hidden_state() is None


class TestHiddenStateDetach:
    """Tests for hidden state detach (TBPTT support)."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_detach_breaks_gradient(self, rnn_type: str) -> None:
        """After detach_hidden_state(), the hidden state should not require grad."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)
        model(obs)

        model.detach_hidden_state()

        state = model.get_hidden_state()
        if rnn_type == "lstm":
            assert not state[0].requires_grad
            assert not state[1].requires_grad
        else:
            assert not state.requires_grad

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_detach_preserves_values(self, rnn_type: str) -> None:
        """Detaching should not change the hidden state values."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)
        model(obs)

        state = model.get_hidden_state()
        if rnn_type == "lstm":
            h_before = state[0].clone()
            c_before = state[1].clone()
        else:
            h_before = state.clone()

        model.detach_hidden_state()

        state_after = model.get_hidden_state()
        if rnn_type == "lstm":
            assert torch.allclose(state_after[0], h_before)
            assert torch.allclose(state_after[1], c_before)
        else:
            assert torch.allclose(state_after, h_before)


class TestSequentialConsistency:
    """Tests that hidden state accumulates context over sequential steps."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_hidden_state_accumulates_context(self, rnn_type: str) -> None:
        """Running multiple steps should change the output compared to a fresh model."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)

        # Fresh output
        output_fresh = model(obs).detach().clone()
        model.reset()

        # Run 3 steps to accumulate hidden state
        for _ in range(3):
            model(obs)
        output_with_context = model(obs).detach()

        assert not torch.allclose(output_fresh, output_with_context, atol=1e-6), (
            "Output should differ after accumulating hidden state context"
        )

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_reset_restores_fresh_behavior(self, rnn_type: str) -> None:
        """After reset(), output should match a fresh start."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)

        # Fresh output
        output_fresh = model(obs).detach().clone()

        # Accumulate context
        for _ in range(5):
            model(obs)

        # Reset and check
        model.reset()
        output_after_reset = model(obs).detach()

        assert torch.allclose(output_fresh, output_after_reset, atol=1e-6), (
            "Output after reset should match fresh output"
        )


class TestRNNModelJITExport:
    """Tests for JIT export fidelity of RNN models."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_jit_export_model(self, rnn_type: str) -> None:
        """JIT export of an RNN model should produce deterministic outputs matching the original."""
        obs = make_obs(1, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            activation="elu",
            rnn_type=rnn_type,
            rnn_hidden_dim=16,
            rnn_num_layers=1,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        original_output = model(obs, stochastic_output=False).detach()

        jit_model = torch.jit.script(model.as_jit())
        jit_model.reset()
        obs_concat = torch.cat([obs[g] for g in model.obs_groups], dim=-1)
        jit_output = jit_model(obs_concat)

        assert torch.allclose(original_output, jit_output, atol=1e-5), f"JIT stochastic export mismatch for {rnn_type}"

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_jit_export_sequential_consistency(self, rnn_type: str) -> None:
        """JIT model should accumulate hidden state across steps, matching original multi-step behavior."""
        obs = make_obs(1, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            activation="elu",
            rnn_type=rnn_type,
            rnn_hidden_dim=16,
            rnn_num_layers=1,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        jit_model = torch.jit.script(model.as_jit())
        jit_model.reset()
        obs_concat = torch.cat([obs[g] for g in model.obs_groups], dim=-1)

        for _ in range(3):
            original_out = model(obs, stochastic_output=False).detach()
            jit_out = jit_model(obs_concat)

        assert torch.allclose(original_out, jit_out, atol=1e-5), f"JIT sequential mismatch for {rnn_type}"
