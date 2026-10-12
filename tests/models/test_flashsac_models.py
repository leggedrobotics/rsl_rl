# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FlashSAC actor and critic models."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import FlashSACActorModel, FlashSACCriticModel

NUM_ENVS = 4
POLICY_DIM = 8
PRIV_DIM = 5
NUM_ACTIONS = 3
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy", "privileged"]}


def _make_obs(num_envs: int = NUM_ENVS) -> TensorDict:
    """Create an observation TensorDict with 'policy' and 'privileged' groups."""
    return TensorDict(
        {
            "policy": torch.randn(num_envs, POLICY_DIM),
            "privileged": torch.randn(num_envs, PRIV_DIM),
        },
        batch_size=[num_envs],
    )


def _make_actor(**kwargs: object) -> tuple[FlashSACActorModel, TensorDict]:
    """Create a FlashSACActorModel and matching observations."""
    obs = _make_obs()
    model = FlashSACActorModel(obs, OBS_GROUPS, "actor", NUM_ACTIONS, num_blocks=2, hidden_dim=32, **kwargs)
    return model, obs


def _make_critic(**kwargs: object) -> tuple[FlashSACCriticModel, TensorDict]:
    """Create a FlashSACCriticModel and matching observations."""
    obs = _make_obs()
    model = FlashSACCriticModel(
        obs, OBS_GROUPS, "critic", action_dim=NUM_ACTIONS, num_blocks=2, hidden_dim=32, num_bins=51, **kwargs
    )
    return model, obs


class TestActorModel:
    """Tests for the actor model surface."""

    def test_obs_dim_from_groups(self) -> None:
        """Actor uses only the 'actor' set; critic sums 'policy' + 'privileged'."""
        actor, _ = _make_actor()
        critic, _ = _make_critic()
        assert actor.obs_dim == POLICY_DIM
        assert critic.obs_dim == POLICY_DIM + PRIV_DIM

    def test_deterministic_forward_is_bounded(self) -> None:
        """forward() returns deterministic tanh(mean) actions in [-1, 1]."""
        actor, obs = _make_actor()
        out = actor(obs)
        assert out.shape == (NUM_ENVS, NUM_ACTIONS)
        assert (out.abs() <= 1.0 + 1e-6).all()

    def test_deterministic_matches_tanh_mean(self) -> None:
        """forward() equals tanh(mean) from mean_std()."""
        actor, obs = _make_actor()
        mean, _ = actor.mean_std(obs, training=False)
        assert torch.allclose(actor(obs), torch.tanh(mean), atol=1e-6)

    def test_sample_returns_logprob(self) -> None:
        """sample() returns actions and a per-sample log-prob."""
        actor, obs = _make_actor()
        actions, log_prob = actor.sample(obs, training=True)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)
        assert log_prob.shape == (NUM_ENVS,)


class TestCriticModel:
    """Tests for the critic model surface."""

    def test_forward_shapes(self) -> None:
        """forward() returns per-ensemble values and per-bin log-probs."""
        critic, obs = _make_critic()
        actions = torch.rand(NUM_ENVS, NUM_ACTIONS) * 2 - 1
        qs, info = critic(obs, actions, training=True)
        assert qs.shape == (2, NUM_ENVS)
        assert info["log_prob"].shape == (2, NUM_ENVS, 51)


class TestObsGroupValidation:
    """Tests for fail-loud observation-group handling."""

    def test_missing_set_raises(self) -> None:
        """A missing observation set raises KeyError."""
        obs = _make_obs()
        with pytest.raises(KeyError):
            FlashSACActorModel(obs, {"critic": ["policy"]}, "actor", NUM_ACTIONS)

    def test_non_1d_obs_raises(self) -> None:
        """A non-1D observation group raises ValueError."""
        obs = TensorDict({"policy": torch.randn(NUM_ENVS, 4, 4)}, batch_size=[NUM_ENVS])
        with pytest.raises(ValueError):
            FlashSACActorModel(obs, {"actor": ["policy"]}, "actor", NUM_ACTIONS)


class TestActorExport:
    """Tests for actor export fidelity."""

    @pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_model(self) -> None:
        """ONNX-exported actor is a valid graph with the expected I/O names."""
        actor, _ = _make_actor()
        actor.eval()
        onnx_model = actor.as_onnx(verbose=False)
        onnx_model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
                dynamo=False,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)
            assert [i.name for i in loaded.graph.input] == ["obs"]
            assert [o.name for o in loaded.graph.output] == ["actions"]

    @pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_matches_eager(self) -> None:
        """The ONNX-export wrapper reproduces the eager deterministic action."""
        actor, obs = _make_actor()
        actor.eval()
        onnx_model = actor.as_onnx(verbose=False)
        onnx_model.eval()
        obs_concat = torch.cat([obs[g] for g in actor.obs_groups], dim=-1)
        assert torch.allclose(actor(obs), onnx_model(obs_concat), atol=1e-6)
