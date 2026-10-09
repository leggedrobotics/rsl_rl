# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the equivariant building blocks.

A wrong symmetry representation produces a network that trains without error and only shows up later as a
bad policy, so the constraint is checked numerically at every level: the representation itself, a single
layer, and the assembled network before and after optimization.
"""

import torch
from tensordict import TensorDict
from types import SimpleNamespace

import pytest

from rsl_rl.modules import (
    EquivariantGaussianDistribution,
    EquivariantLinear,
    EquivariantMLP,
    SignedPermutation,
    symmetry_cfg_from_augmentation,
)

# A toy four-joint biped: joints are (left_a, right_a, left_b, right_b). The "b" pair flips sign under the
# reflection, as a roll or yaw joint would; the "a" pair does not, as a pitch joint would.
ACT_PERM = [1, 0, 3, 2]
ACT_SIGN = [1.0, 1.0, -1.0, -1.0]
# Observation: linear velocity (3), angular velocity (3, a pseudovector), then the four joint positions.
OBS_PERM = [0, 1, 2, 3, 4, 5] + [6 + i for i in ACT_PERM]
OBS_SIGN = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, *ACT_SIGN]
OBS_DIM = len(OBS_PERM)
ACT_DIM = len(ACT_PERM)
HIDDEN = [32, 16]
BATCH = 64


@pytest.fixture
def reps() -> tuple[SignedPermutation, SignedPermutation]:
    """Return the observation and action representations of the toy robot."""
    return SignedPermutation(OBS_PERM, OBS_SIGN), SignedPermutation(ACT_PERM, ACT_SIGN)


class TestSignedPermutation:
    """Tests for ``SignedPermutation``."""

    def test_is_involution(self, reps: tuple[SignedPermutation, SignedPermutation]) -> None:
        """Applying the transformation twice returns the input."""
        rep_in, rep_out = reps
        obs = torch.randn(BATCH, OBS_DIM)
        act = torch.randn(BATCH, ACT_DIM)
        assert torch.allclose(rep_in(rep_in(obs)), obs)
        assert torch.allclose(rep_out(rep_out(act)), act)

    @pytest.mark.parametrize(
        ("perm", "sign"),
        [
            ([1, 2, 0], None),  # permutation is not an involution
            ([0, 1], [1.0, 0.5]),  # sign is not +-1
            ([1, 0], [1.0, -1.0]),  # signed map is not an involution
            ([0, 0], None),  # not a permutation
        ],
    )
    def test_rejects_invalid_representation(self, perm: list[int], sign: list[float] | None) -> None:
        """An invalid representation is rejected at construction, not silently used."""
        with pytest.raises(ValueError):
            SignedPermutation(perm, sign)

    def test_regular_requires_even_dimension(self) -> None:
        """The regular representation needs a dimension divisible by the group order."""
        with pytest.raises(ValueError):
            SignedPermutation.regular(5)


class TestEquivariantLinear:
    """Tests for ``EquivariantLinear``."""

    @pytest.mark.parametrize(
        "make_reps",
        [
            lambda: (SignedPermutation(OBS_PERM, OBS_SIGN), SignedPermutation.regular(32)),
            lambda: (SignedPermutation.regular(32), SignedPermutation.regular(16)),
            lambda: (SignedPermutation.regular(16), SignedPermutation(ACT_PERM, ACT_SIGN)),
            lambda: (SignedPermutation.regular(16), SignedPermutation.identity(1)),
        ],
        ids=["obs-to-hidden", "hidden-to-hidden", "hidden-to-action", "hidden-to-value"],
    )
    def test_satisfies_constraint(self, make_reps: callable) -> None:
        """The projected weight satisfies W M_in == M_out W exactly."""
        rep_in, rep_out = make_reps()
        layer = EquivariantLinear(rep_in, rep_out)
        weight, _ = layer.equivariant_weight()
        m_in = torch.zeros(len(rep_in), len(rep_in))
        m_in[torch.arange(len(rep_in)), rep_in.perm] = rep_in.sign
        m_out = torch.zeros(len(rep_out), len(rep_out))
        m_out[torch.arange(len(rep_out)), rep_out.perm] = rep_out.sign
        assert torch.allclose(weight @ m_in, m_out @ weight, atol=1e-6)

    def test_fold_preserves_the_function(self, reps: tuple[SignedPermutation, SignedPermutation]) -> None:
        """Folding the projection into a plain linear layer does not change the output."""
        rep_in, rep_out = reps
        layer = EquivariantLinear(rep_in, rep_out)
        obs = torch.randn(BATCH, OBS_DIM)
        with torch.no_grad():
            assert torch.allclose(layer(obs), layer.fold()(obs), atol=1e-6)

    def test_initial_scale_matches_plain_linear(self) -> None:
        """The projected weight starts at the scale of an nn.Linear, so deep networks do not start shrunk."""
        torch.manual_seed(0)
        layer = EquivariantLinear(SignedPermutation.regular(512), SignedPermutation.regular(512))
        weight, _ = layer.equivariant_weight()
        plain = torch.nn.Linear(512, 512).weight
        assert weight.std().item() == pytest.approx(plain.std().item(), rel=0.05)


class TestEquivariantMLP:
    """Tests for ``EquivariantMLP``."""

    def test_is_equivariant(self, reps: tuple[SignedPermutation, SignedPermutation]) -> None:
        """A mirrored input produces the mirrored output."""
        rep_in, rep_out = reps
        net = EquivariantMLP(rep_in, rep_out, HIDDEN)
        obs = torch.randn(BATCH, OBS_DIM)
        with torch.no_grad():
            assert torch.allclose(rep_out(net(obs)), net(rep_in(obs)), atol=1e-5)

    def test_trivial_output_is_invariant(self, reps: tuple[SignedPermutation, SignedPermutation]) -> None:
        """With the trivial output representation the network is invariant, as a critic needs."""
        rep_in, _ = reps
        net = EquivariantMLP(rep_in, SignedPermutation.identity(1), HIDDEN)
        obs = torch.randn(BATCH, OBS_DIM)
        with torch.no_grad():
            assert torch.allclose(net(obs), net(rep_in(obs)), atol=1e-5)

    def test_equivariance_survives_optimization(self, reps: tuple[SignedPermutation, SignedPermutation]) -> None:
        """The constraint is structural, so gradient steps cannot trade it away."""
        rep_in, rep_out = reps
        net = EquivariantMLP(rep_in, rep_out, HIDDEN)
        obs = torch.randn(BATCH, OBS_DIM)
        target = torch.randn(BATCH, ACT_DIM)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(50):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(net(obs), target).backward()
            optimizer.step()
        with torch.no_grad():
            assert torch.allclose(rep_out(net(obs)), net(rep_in(obs)), atol=1e-5)

    def test_fold_preserves_the_function(self, reps: tuple[SignedPermutation, SignedPermutation]) -> None:
        """The folded network computes the same function as the projected one."""
        rep_in, rep_out = reps
        net = EquivariantMLP(rep_in, rep_out, HIDDEN)
        obs = torch.randn(BATCH, OBS_DIM)
        with torch.no_grad():
            assert torch.allclose(net(obs), net.fold()(obs), atol=1e-6)


class TestEquivariantGaussianDistribution:
    """Tests for ``EquivariantGaussianDistribution``."""

    def test_std_is_shared_between_mirrored_pairs(self) -> None:
        """Mirrored outputs get the same exploration noise, so sampling stays equivariant."""
        dist = EquivariantGaussianDistribution(ACT_DIM, perm=ACT_PERM)
        with torch.no_grad():
            dist.std_param.copy_(torch.tensor([0.8, 0.2, 1.4, 0.6]))
            dist.update(torch.zeros(8, ACT_DIM))
        std = dist.std[0]
        assert torch.allclose(std[0], std[1])
        assert torch.allclose(std[2], std[3])

    def test_stored_parameter_is_untouched(self) -> None:
        """Symmetrization happens at use time, leaving the parameter and optimizer state alone."""
        dist = EquivariantGaussianDistribution(ACT_DIM, perm=ACT_PERM)
        raw = torch.tensor([0.8, 0.2, 1.4, 0.6])
        with torch.no_grad():
            dist.std_param.copy_(raw)
            dist.update(torch.zeros(8, ACT_DIM))
        assert torch.allclose(dist.std_param, raw)


def augment(env: object, obs: TensorDict | None = None, actions: torch.Tensor | None = None) -> tuple:
    """Toy augmentation function in the style of the symmetry extension.

    Like the Isaac Lab functions, it reads the environment, only mirrors the ``policy`` group, and returns the
    original samples followed by the mirrored ones.
    """
    rep_in, rep_out = SignedPermutation(OBS_PERM, OBS_SIGN), SignedPermutation(ACT_PERM, ACT_SIGN)
    obs_aug = actions_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        obs_aug["policy"][obs.batch_size[0] :] = rep_in(obs["policy"]) * env.scale
    if actions is not None:
        actions_aug = torch.cat([actions, rep_out(actions)])
    return obs_aug, actions_aug


ENV = SimpleNamespace(scale=1.0)


def make_aug_obs() -> TensorDict:
    """Create observations with a mirrored group, an untouched group, and a 2D group outside the model."""
    return TensorDict(
        {
            "policy": torch.randn(BATCH, OBS_DIM),
            "critic": torch.randn(BATCH, OBS_DIM),
            "image": torch.randn(BATCH, 1, 4, 4),
        },
        batch_size=[BATCH],
    )


class TestSymmetryCfgFromAugmentation:
    """Tests for ``symmetry_cfg_from_augmentation``."""

    def test_recovers_representations(self) -> None:
        """The derived representations are exactly the ones the function applies."""
        cfg = symmetry_cfg_from_augmentation(augment, ENV, make_aug_obs(), ["policy"], ACT_DIM)
        assert cfg == {"obs": {"perm": OBS_PERM, "sign": OBS_SIGN}, "output": {"perm": ACT_PERM, "sign": ACT_SIGN}}

    def test_critic_has_no_output(self) -> None:
        """Without the action dimension only the observation representation is derived, as for a critic."""
        cfg = symmetry_cfg_from_augmentation(augment, ENV, make_aug_obs(), ["policy"])
        assert set(cfg) == {"obs"}

    def test_matches_function_on_data(self) -> None:
        """Applying the derived representation equals applying the function."""
        obs = make_aug_obs()
        cfg = symmetry_cfg_from_augmentation(augment, ENV, obs, ["policy"], ACT_DIM)
        obs_aug, _ = augment(ENV, obs)
        assert torch.equal(SignedPermutation(**cfg["obs"])(obs["policy"]), obs_aug["policy"][BATCH:])

    def test_rejects_unmirrored_group(self) -> None:
        """A group the function leaves unchanged is an error, not a silently unconstrained model."""
        with pytest.raises(ValueError, match="unchanged"):
            symmetry_cfg_from_augmentation(augment, ENV, make_aug_obs(), ["critic"])

    @pytest.mark.parametrize("scale", [2.0, 0.5])
    def test_rejects_non_signed_permutation(self, scale: float) -> None:
        """A function that scales entries is not a signed permutation."""
        with pytest.raises(ValueError, match="signed permutation"):
            symmetry_cfg_from_augmentation(augment, SimpleNamespace(scale=scale), make_aug_obs(), ["policy"])

    def test_rejects_affine_function(self) -> None:
        """A function with an offset is not linear."""

        def shifted(env: object, obs: TensorDict | None = None, actions: torch.Tensor | None = None) -> tuple:
            obs_aug, actions_aug = augment(env, obs, actions)
            obs_aug["policy"][obs.batch_size[0] :] += 1.0
            return obs_aug, actions_aug

        with pytest.raises(ValueError, match="zero"):
            symmetry_cfg_from_augmentation(shifted, ENV, make_aug_obs(), ["policy"])

    def test_rejects_invalid_aug_index(self) -> None:
        """The original slice and slices beyond the augmentation are rejected."""
        for aug_index in (0, 2):
            with pytest.raises(ValueError, match="aug_index"):
                symmetry_cfg_from_augmentation(augment, ENV, make_aug_obs(), ["policy"], aug_index=aug_index)
