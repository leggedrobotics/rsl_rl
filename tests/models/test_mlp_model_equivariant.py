# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the EquivariantMLPModel."""

from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.models import EquivariantMLPModel
from rsl_rl.modules import EquivariantLinear, SignedPermutation

ACT_PERM = [1, 0, 3, 2]
ACT_SIGN = [1.0, 1.0, -1.0, -1.0]
OBS_PERM = [0, 1, 2, 3, 4, 5] + [6 + i for i in ACT_PERM]
OBS_SIGN = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, *ACT_SIGN]
OBS_DIM = len(OBS_PERM)
ACT_DIM = len(ACT_PERM)
HIDDEN = [32, 16]
NUM_ENVS = 16

OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}
OBS_CFG = {"perm": OBS_PERM, "sign": OBS_SIGN}
ACT_CFG = {"perm": ACT_PERM, "sign": ACT_SIGN}


def make_obs() -> TensorDict:
    """Create an observation TensorDict for the toy robot."""
    return TensorDict({"policy": torch.randn(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS])


def mirror_obs(obs: TensorDict) -> TensorDict:
    """Return the reflected observation."""
    rep = SignedPermutation(OBS_PERM, OBS_SIGN)
    return TensorDict({"policy": rep(obs["policy"])}, batch_size=obs.batch_size)


class TestEquivariantMLPModel:
    """Tests for ``EquivariantMLPModel``."""

    def test_actor_is_equivariant(self) -> None:
        """A mirrored observation produces the mirrored action."""
        obs = make_obs()
        model = EquivariantMLPModel(
            obs,
            OBS_GROUPS,
            "actor",
            ACT_DIM,
            hidden_dims=HIDDEN,
            equivariance_cfg={"obs": OBS_CFG, "output": ACT_CFG},
        )
        rep_out = SignedPermutation(ACT_PERM, ACT_SIGN)
        with torch.no_grad():
            assert torch.allclose(rep_out(model(obs)), model(mirror_obs(obs)), atol=1e-5)

    def test_critic_is_invariant(self) -> None:
        """Omitting the output representation makes the value invariant."""
        obs = make_obs()
        model = EquivariantMLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=HIDDEN, equivariance_cfg={"obs": OBS_CFG})
        with torch.no_grad():
            assert torch.allclose(model(obs), model(mirror_obs(obs)), atol=1e-5)

    def test_stochastic_output_is_equivariant(self) -> None:
        """With the equivariant distribution the mean action is still equivariant."""
        obs = make_obs()
        model = EquivariantMLPModel(
            obs,
            OBS_GROUPS,
            "actor",
            ACT_DIM,
            hidden_dims=HIDDEN,
            equivariance_cfg={"obs": OBS_CFG, "output": ACT_CFG},
            distribution_cfg={"class_name": "EquivariantGaussianDistribution", "perm": ACT_PERM},
        )
        rep_out = SignedPermutation(ACT_PERM, ACT_SIGN)
        with torch.no_grad():
            assert torch.allclose(rep_out(model(obs)), model(mirror_obs(obs)), atol=1e-5)

    def test_requires_equivariance_cfg(self) -> None:
        """The model cannot be built without a symmetry representation."""
        with pytest.raises(ValueError, match="equivariance_cfg"):
            EquivariantMLPModel(make_obs(), OBS_GROUPS, "actor", ACT_DIM, hidden_dims=HIDDEN)

    def test_normalized_model_stays_equivariant(self) -> None:
        """With asymmetric data, the normalization statistics stay symmetric and the model equivariant."""
        obs = make_obs()
        model = EquivariantMLPModel(
            obs,
            OBS_GROUPS,
            "actor",
            ACT_DIM,
            hidden_dims=HIDDEN,
            obs_normalization=True,
            equivariance_cfg={"obs": OBS_CFG, "output": ACT_CFG},
        )
        # Data that is strongly biased towards one side of the robot.
        for _ in range(5):
            biased = make_obs()
            biased["policy"] = biased["policy"] * torch.linspace(0.5, 3.0, OBS_DIM) + torch.linspace(-2.0, 2.0, OBS_DIM)
            model.update_normalization(biased)
        rep_in = SignedPermutation(OBS_PERM, OBS_SIGN)
        rep_out = SignedPermutation(ACT_PERM, ACT_SIGN)
        normalizer = model.obs_normalizer
        assert torch.equal(rep_in(normalizer.mean), normalizer.mean)
        assert torch.equal(normalizer.std[rep_in.perm], normalizer.std)
        with torch.no_grad():
            assert torch.allclose(rep_out(model(obs)), model(mirror_obs(obs)), atol=1e-5)

    def test_rejects_mismatched_representation(self) -> None:
        """A representation whose size does not match the observation is an error, not a silent reshape."""
        with pytest.raises(ValueError, match="dimensions"):
            EquivariantMLPModel(
                make_obs(), OBS_GROUPS, "actor", ACT_DIM, hidden_dims=HIDDEN, equivariance_cfg={"obs": ACT_CFG}
            )

    def test_rejects_structured_distribution(self) -> None:
        """Distributions with a structured MLP output are reported as unsupported, not as a size mismatch."""
        with pytest.raises(NotImplementedError, match="HeteroscedasticGaussianDistribution"):
            EquivariantMLPModel(
                make_obs(),
                OBS_GROUPS,
                "actor",
                ACT_DIM,
                hidden_dims=HIDDEN,
                distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution"},
                equivariance_cfg={"obs": OBS_CFG, "output": ACT_CFG},
            )


def make_trained_actor() -> EquivariantMLPModel:
    """Create a stochastic actor with normalization statistics learned from asymmetric data."""
    model = EquivariantMLPModel(
        make_obs(),
        OBS_GROUPS,
        "actor",
        ACT_DIM,
        hidden_dims=HIDDEN,
        obs_normalization=True,
        distribution_cfg={"class_name": "EquivariantGaussianDistribution", "perm": ACT_PERM},
        equivariance_cfg={"obs": OBS_CFG, "output": ACT_CFG},
    )
    for _ in range(5):
        biased = make_obs()
        biased["policy"] = biased["policy"] + torch.linspace(-2.0, 2.0, OBS_DIM)
        model.update_normalization(biased)
    model.eval()
    return model


class TestEquivariantMLPModelExport:
    """Tests for the export of ``EquivariantMLPModel``."""

    def test_jit_export_is_folded(self) -> None:
        """The JIT export uses a plain MLP and matches the original model."""
        model = make_trained_actor()
        exported = model.as_jit()
        assert not any(isinstance(m, EquivariantLinear) for m in exported.modules())
        obs = make_obs()
        jit_model = torch.jit.script(exported)
        with torch.no_grad():
            assert torch.allclose(model(obs), jit_model(obs["policy"]), atol=1e-5)

    @pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
    @pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
    def test_onnx_export_is_folded(self) -> None:
        """The ONNX export uses a plain MLP and is a valid graph."""
        model = make_trained_actor()
        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()
        assert not any(isinstance(m, EquivariantLinear) for m in onnx_model.modules())
        with torch.no_grad():
            obs = make_obs()
            assert torch.allclose(model(obs), onnx_model(obs["policy"]), atol=1e-5)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            onnx.checker.check_model(onnx.load(f.name))
