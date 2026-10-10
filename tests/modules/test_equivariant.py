# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the equivariant building blocks.

A wrong symmetry representation produces a network that trains without error and only shows up later as a
bad policy, so the constraint is checked numerically at every level: the representation itself, a single
layer, and the assembled network before and after optimization.
"""

from __future__ import annotations

import torch

import pytest

from rsl_rl.modules import EquivariantGaussianDistribution, EquivariantLinear, EquivariantMLP, SignedPermutation

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
