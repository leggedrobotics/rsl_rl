# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the EquivariantMLPModel."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models import EquivariantMLPModel
from rsl_rl.modules import SignedPermutation

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
            symmetry_cfg={"obs": OBS_CFG, "output": ACT_CFG},
        )
        rep_out = SignedPermutation(ACT_PERM, ACT_SIGN)
        with torch.no_grad():
            assert torch.allclose(rep_out(model(obs)), model(mirror_obs(obs)), atol=1e-5)

    def test_critic_is_invariant(self) -> None:
        """Omitting the output representation makes the value invariant."""
        obs = make_obs()
        model = EquivariantMLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=HIDDEN, symmetry_cfg={"obs": OBS_CFG})
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
            symmetry_cfg={"obs": OBS_CFG, "output": ACT_CFG},
            distribution_cfg={"class_name": "EquivariantGaussianDistribution", "perm": ACT_PERM},
        )
        rep_out = SignedPermutation(ACT_PERM, ACT_SIGN)
        with torch.no_grad():
            assert torch.allclose(rep_out(model(obs)), model(mirror_obs(obs)), atol=1e-5)

    def test_requires_symmetry_cfg(self) -> None:
        """The model cannot be built without a symmetry representation."""
        with pytest.raises(ValueError, match="symmetry_cfg"):
            EquivariantMLPModel(make_obs(), OBS_GROUPS, "actor", ACT_DIM, hidden_dims=HIDDEN)

    def test_rejects_observation_normalization(self) -> None:
        """Running observation statistics are not symmetric, so normalization is refused."""
        with pytest.raises(ValueError, match="obs_normalization"):
            EquivariantMLPModel(
                make_obs(),
                OBS_GROUPS,
                "actor",
                ACT_DIM,
                hidden_dims=HIDDEN,
                obs_normalization=True,
                symmetry_cfg={"obs": OBS_CFG},
            )

    def test_rejects_mismatched_representation(self) -> None:
        """A representation whose size does not match the observation is an error, not a silent reshape."""
        with pytest.raises(ValueError, match="dimensions"):
            EquivariantMLPModel(
                make_obs(), OBS_GROUPS, "actor", ACT_DIM, hidden_dims=HIDDEN, symmetry_cfg={"obs": ACT_CFG}
            )
