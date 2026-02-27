# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLPModel."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import MLPModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_mlp_model(stochastic: bool = False, obs_set: str = "actor", **kwargs: object) -> tuple[MLPModel, TensorDict]:
    """Create an MLPModel and matching observations for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    if stochastic:
        defaults["distribution_cfg"] = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }
    defaults.update(kwargs)
    output_dim = NUM_ACTIONS if stochastic else 1
    model = MLPModel(obs, OBS_GROUPS, obs_set, output_dim, **defaults)
    return model, obs


class TestMLPModelModes:
    """Tests for stochastic vs. deterministic forward pass behavior."""

    def test_deterministic_returns_mean(self) -> None:
        """forward() should return the distribution mean."""
        actor, obs = _make_mlp_model(stochastic=True)
        actor.eval()

        det_output = actor(obs)

        actor(obs, stochastic_output=True)
        mean_output = actor.output_mean

        assert torch.allclose(det_output, mean_output, atol=1e-6)

    def test_stochastic_differs_from_deterministic(self) -> None:
        """Stochastic samples should not exactly equal the mean (with overwhelming probability)."""
        actor, obs = _make_mlp_model(stochastic=True)

        torch.manual_seed(42)
        stochastic_output = actor(obs, stochastic_output=True)
        det_output = actor(obs, stochastic_output=False)

        assert not torch.allclose(stochastic_output, det_output, atol=1e-6), (
            "Stochastic output should differ from deterministic mean"
        )

    def test_no_distribution_returns_raw_mlp(self) -> None:
        """Without distribution_cfg, forward should return raw MLP output."""
        critic, obs = _make_mlp_model(stochastic=False, obs_set="critic")

        output = critic(obs)
        latent = critic.get_latent(obs)
        expected = critic.mlp(latent)
        assert torch.allclose(output, expected)


class TestMLPModelNormalization:
    """Tests for observation normalization integration."""

    def test_normalization_changes_output(self) -> None:
        """A model with obs_normalization should produce different outputs after normalization stats update."""
        model, obs = _make_mlp_model(stochastic=False, obs_set="critic", obs_normalization=True)
        model.train()

        output_before = model(obs).detach().clone()

        shifted_obs = make_obs(NUM_ENVS, OBS_DIM)
        shifted_obs["policy"] = shifted_obs["policy"] + 100.0
        for _ in range(50):
            model.update_normalization(shifted_obs)

        output_after = model(obs).detach()
        assert not torch.allclose(output_before, output_after, atol=1e-3), (
            "Output should change after normalization stats update"
        )


class TestObsGroupConcatenation:
    """Tests for observation group concatenation order."""

    def test_concatenation_order_is_preserved(self) -> None:
        """get_latent should concatenate obs groups in the declared order."""
        obs = TensorDict(
            {
                "group_a": torch.ones(2, 3),
                "group_b": torch.ones(2, 4) * 2,
            },
            batch_size=[2],
        )
        obs_groups = {"actor": ["group_a", "group_b"], "critic": ["group_a"]}
        model = MLPModel(obs, obs_groups, "actor", 1, hidden_dims=[8])

        latent = model.get_latent(obs)
        # First 3 dims should be 1.0 (group_a), next 4 should be 2.0 (group_b)
        assert torch.allclose(latent[:, :3], torch.ones(2, 3))
        assert torch.allclose(latent[:, 3:], torch.ones(2, 4) * 2)

    def test_reversed_order_gives_different_latent(self) -> None:
        """Swapping the obs group order should change the latent representation."""
        obs = TensorDict(
            {
                "group_a": torch.ones(2, 3),
                "group_b": torch.ones(2, 3) * 5,
            },
            batch_size=[2],
        )

        model_ab = MLPModel(obs, {"actor": ["group_a", "group_b"]}, "actor", 1, hidden_dims=[8])
        model_ba = MLPModel(obs, {"actor": ["group_b", "group_a"]}, "actor", 1, hidden_dims=[8])

        latent_ab = model_ab.get_latent(obs)
        latent_ba = model_ba.get_latent(obs)

        assert not torch.allclose(latent_ab, latent_ba), "Different obs group orders should produce different latents"


class TestMLPModelExport:
    """Tests for JIT export fidelity."""

    def test_jit_export_model(self) -> None:
        """JIT-exported model should produce deterministic outputs matching the original."""
        actor, obs = _make_mlp_model(stochastic=True)
        actor.eval()

        original_output = actor(obs).detach()

        jit_model = torch.jit.script(actor.as_jit())
        obs_concat = torch.cat([obs[g] for g in actor.obs_groups], dim=-1)
        jit_output = jit_model(obs_concat)

        assert torch.allclose(original_output, jit_output, atol=1e-5), "JIT stochastic export should match original"

    def test_jit_export_with_normalization(self) -> None:
        """JIT export should preserve trained normalization statistics."""
        model, obs = _make_mlp_model(stochastic=True, obs_normalization=True)
        model.train()

        for _ in range(50):
            shifted_obs = TensorDict(
                {"policy": torch.randn(NUM_ENVS, OBS_DIM) + 5.0},
                batch_size=[NUM_ENVS],
            )
            model.update_normalization(shifted_obs)

        model.eval()
        jit_model = torch.jit.script(model.as_jit())

        obs_concat = torch.cat([obs[g] for g in model.obs_groups], dim=-1)
        original_output = model(obs)
        jit_output = jit_model(obs_concat)

        assert torch.allclose(original_output, jit_output, atol=1e-5)

    @pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_model(self) -> None:
        """ONNX-exported MLP model should be a valid ONNX graph with correct I/O names."""
        actor, _obs = _make_mlp_model(stochastic=True)
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
                dynamic_axes={},
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)

            assert [i.name for i in loaded.graph.input] == ["obs"]
            assert [o.name for o in loaded.graph.output] == ["actions"]

    @pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_with_normalization(self) -> None:
        """ONNX export should produce a valid graph when obs normalization is enabled."""
        model, _obs = _make_mlp_model(stochastic=True, obs_normalization=True)
        model.train()

        for _ in range(50):
            shifted_obs = TensorDict(
                {"policy": torch.randn(NUM_ENVS, OBS_DIM) + 5.0},
                batch_size=[NUM_ENVS],
            )
            model.update_normalization(shifted_obs)

        model.eval()
        onnx_model = model.as_onnx(verbose=False)
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
                dynamic_axes={},
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)
