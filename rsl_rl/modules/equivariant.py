# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

__all__ = [
    "EquivariantGaussianDistribution",
    "EquivariantLinear",
    "EquivariantMLP",
    "SignedPermutation",
    "SymmetricEmpiricalNormalization",
]


class SignedPermutation:
    """A signed permutation acting on the last dimension of a tensor.

    The transformation is ``M x = sign * x[perm]``. It represents one element of a symmetry group of the
    robot, most commonly the left-right reflection of a legged platform: the two sides swap, and the joints
    whose axis flips under the reflection (roll and yaw, but not pitch) additionally change sign.

    Only involutions are supported, i.e. ``M(M x) == x``, which covers reflections. This is checked on
    construction, since a wrong sign or a wrong index pairing produces a transformation that still runs but
    silently trains the wrong function.
    """

    def __init__(self, perm: torch.Tensor | list[int], sign: torch.Tensor | list[float] | None = None) -> None:
        """Initialize the signed permutation.

        Args:
            perm: Index permutation, of length equal to the dimension it acts on.
            sign: Per-index sign, either ``+1`` or ``-1``. Defaults to all ``+1``.

        Raises:
            ValueError: If the permutation is not an involution, if the signs are not +-1, or if the signed
                permutation as a whole is not an involution.
        """
        perm = torch.as_tensor(perm, dtype=torch.long)
        sign = torch.ones(len(perm)) if sign is None else torch.as_tensor(sign, dtype=torch.float)
        if perm.ndim != 1 or sign.shape != perm.shape:
            raise ValueError(f"perm and sign must be 1D of equal length, got {tuple(perm.shape)}, {tuple(sign.shape)}")
        if sorted(perm.tolist()) != list(range(len(perm))):
            raise ValueError("perm must be a permutation of range(dim)")
        if not torch.all(perm[perm] == torch.arange(len(perm))):
            raise ValueError("perm must be an involution (perm[perm] == identity)")
        if not torch.all(torch.isclose(sign.abs(), torch.ones_like(sign))):
            raise ValueError("sign entries must be +1 or -1")
        if not torch.all(torch.isclose(sign * sign[perm], torch.ones_like(sign))):
            raise ValueError("signed permutation must be an involution (sign[i] * sign[perm[i]] == 1)")
        self.perm = perm
        self.sign = sign

    def __len__(self) -> int:
        """Return the dimension the transformation acts on."""
        return len(self.perm)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the last dimension of the input.

        Args:
            x: Tensor whose last dimension matches this representation.

        Returns:
            The transformed tensor.
        """
        return x.index_select(-1, self.perm.to(x.device)) * self.sign.to(x.device, x.dtype)

    @staticmethod
    def identity(dim: int) -> SignedPermutation:
        """Return the trivial representation, under which ``M`` acts as the identity.

        A model whose output carries this representation is invariant rather than equivariant, which is
        what a critic value needs.

        Args:
            dim: Dimension the representation acts on.

        Returns:
            The identity representation.
        """
        return SignedPermutation(torch.arange(dim))

    @staticmethod
    def regular(dim: int) -> SignedPermutation:
        """Return the regular representation of the reflection group, used on hidden layers.

        The units are split into two halves that are exchanged, with no sign flips. Because it is a pure
        permutation, any pointwise activation commutes with it, so equivariance survives the nonlinearities.

        Args:
            dim: Hidden dimension, which must be even.

        Returns:
            The regular representation.

        Raises:
            ValueError: If ``dim`` is odd.
        """
        if dim % 2 != 0:
            raise ValueError(f"hidden dimension must be even, got {dim}")
        half = dim // 2
        return SignedPermutation(torch.cat([torch.arange(half, dim), torch.arange(0, half)]))


class EquivariantLinear(nn.Module):
    r"""A linear layer constrained to be equivariant, i.e. :math:`W M_\text{in} = M_\text{out} W`.

    The weight is projected onto the equivariant subspace on every forward pass,

    .. math::
        W_\text{eq} = \tfrac{1}{2}\left(W + M_\text{out} W M_\text{in}\right), \qquad
        b_\text{eq} = \tfrac{1}{2}\left(b + M_\text{out} b\right),

    which satisfies the constraint exactly because both representations are involutions. Gradients flow to
    the unconstrained parameter, so the effective weight is equivariant at every step of training and not
    only at convergence. This is the difference from a symmetry loss, which can be traded away against the
    task reward.
    """

    def __init__(self, rep_in: SignedPermutation, rep_out: SignedPermutation) -> None:
        """Initialize the layer.

        Args:
            rep_in: Representation acting on the input.
            rep_out: Representation acting on the output.
        """
        super().__init__()
        self.register_buffer("p_in", rep_in.perm)
        self.register_buffer("s_in", rep_in.sign)
        self.register_buffer("p_out", rep_out.perm)
        self.register_buffer("s_out", rep_out.sign)
        self.weight = nn.Parameter(torch.empty(len(rep_out), len(rep_in)))
        self.bias = nn.Parameter(torch.zeros(len(rep_out)))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # The projection averages each entry with its independently initialized mirror partner, which halves its
        # variance. Compensate so the projected weight starts at the scale of a plain nn.Linear; entries that are
        # their own partner are left unchanged by the projection (or zeroed by it) and need no correction.
        self_paired = (rep_out.perm.unsqueeze(1) == torch.arange(len(rep_out)).unsqueeze(1)) & (
            rep_in.perm.unsqueeze(0) == torch.arange(len(rep_in)).unsqueeze(0)
        )
        with torch.no_grad():
            self.weight[~self_paired] *= 2**0.5

    @property
    def in_features(self) -> int:
        """Return the input dimension."""
        return len(self.p_in)

    @property
    def out_features(self) -> int:
        """Return the output dimension."""
        return len(self.p_out)

    def equivariant_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the weight and bias projected onto the equivariant subspace.

        Returns:
            The projected weight and bias.
        """
        # M_out W permutes and signs the rows; (M_out W) M_in permutes and signs the columns. Both
        # representations are involutions, so sign[perm[j]] == sign[j] and the column step uses s_in directly.
        mw = self.weight.index_select(0, self.p_out) * self.s_out.unsqueeze(1)
        mwm = mw.index_select(1, self.p_in) * self.s_in.unsqueeze(0)
        weight = 0.5 * (self.weight + mwm)
        bias = 0.5 * (self.bias + self.bias.index_select(0, self.p_out) * self.s_out)
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projected weight to the input."""
        weight, bias = self.equivariant_weight()
        return torch.nn.functional.linear(x, weight, bias)

    def fold(self) -> nn.Linear:
        """Return a plain :class:`torch.nn.Linear` computing the same function.

        The projection is a fixed linear map, so a trained equivariant layer is exactly equal to an ordinary
        layer with the projected weight baked in. It is used for export to drop the per-forward index
        operations.
        """
        weight, bias = self.equivariant_weight()
        layer = nn.Linear(self.in_features, self.out_features, device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        return layer

    def extra_repr(self) -> str:
        """Return the layer description shown by ``print``."""
        return f"in_features={self.in_features}, out_features={self.out_features}, equivariant=True"


class EquivariantMLP(nn.Sequential):
    """An MLP whose every linear layer is equivariant.

    The hidden layers carry the regular representation of the group, so the activations commute with the
    symmetry and equivariance is preserved end to end. Drop-in replacement for :class:`~rsl_rl.modules.mlp.MLP`
    inside a model.
    """

    def __init__(
        self,
        rep_in: SignedPermutation,
        rep_out: SignedPermutation,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
    ) -> None:
        """Initialize the equivariant MLP.

        Args:
            rep_in: Representation acting on the input.
            rep_out: Representation acting on the output. Use
                :meth:`SignedPermutation.identity` for an invariant output, e.g. a critic value.
            hidden_dims: Hidden layer dimensions. Each must be even.
            activation: Activation function between layers.
        """
        activation_mod = resolve_nn_activation(activation)
        reps = [rep_in] + [SignedPermutation.regular(dim) for dim in hidden_dims] + [rep_out]
        layers: list[nn.Module] = []
        for index in range(len(reps) - 1):
            layers.append(EquivariantLinear(reps[index], reps[index + 1]))
            if index < len(reps) - 2:
                layers.append(activation_mod)
        super().__init__(*layers)

    def fold(self) -> nn.Sequential:
        """Return an ordinary :class:`torch.nn.Sequential` computing the same function."""
        return nn.Sequential(*[m.fold() if isinstance(m, EquivariantLinear) else m for m in self])


class SymmetricEmpiricalNormalization(EmpiricalNormalization):
    """Empirical normalization whose statistics are symmetric, so that it commutes with the symmetry.

    Normalization satisfies ``N(M x) = M N(x)`` only if mirrored entries share their standard deviation and the
    mean satisfies ``mean == M mean``. Running statistics of real data violate both as soon as the policy or the
    environment is slightly asymmetric. This normalizer therefore learns the statistics of the data together with
    its mirror image, and projects them onto the symmetric subspace after every update, so the constraint holds
    exactly rather than up to the floating-point error of the running average.
    """

    def __init__(self, rep: SignedPermutation, eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize the normalizer.

        Args:
            rep: Representation acting on the normalized values.
            eps: Small value for stability.
            until: If specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        # Every update also learns the mirrored batch, which doubles the sample count.
        super().__init__(len(rep), eps, None if until is None else 2 * until)
        self.register_buffer("perm", rep.perm)
        self.register_buffer("sign", rep.sign)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn the input values and their mirror image."""
        super().update(torch.cat([x, x.index_select(-1, self.perm) * self.sign], dim=0))
        self._mean.copy_(0.5 * (self._mean + self._mean.index_select(-1, self.perm) * self.sign))
        self._var.copy_(0.5 * (self._var + self._var.index_select(-1, self.perm)))
        self._std = torch.sqrt(self._var)


class EquivariantGaussianDistribution(GaussianDistribution):
    r"""Gaussian whose standard deviation is shared between mirrored output pairs.

    An equivariant network makes the *mean* action equivariant, but the sampled action is only equivariant
    if the noise is too. With an unconstrained per-dimension standard deviation the left and right joints
    would receive different amounts of exploration noise, which reintroduces the asymmetry the model was
    meant to remove.

    The standard deviation is a scale, so only the permutation of the representation is applied and the
    signs are ignored:

    .. math::
        \sigma_\text{sym} = \tfrac{1}{2}\left(\sigma + \sigma[\text{perm}]\right)

    The symmetrization is applied to the value used by the distribution, not to the stored parameter, so
    the optimizer state is left untouched.
    """

    def __init__(self, output_dim: int, perm: torch.Tensor | list[int], **kwargs: Any) -> None:
        """Initialize the distribution.

        Args:
            output_dim: Dimension of the action/output space.
            perm: Index permutation pairing mirrored outputs, the same one used for the model's output
                representation.
            **kwargs: Forwarded to :class:`~rsl_rl.modules.distribution.GaussianDistribution`.
        """
        super().__init__(output_dim, **kwargs)
        # Validated through SignedPermutation so a non-involutive pairing is rejected early.
        self.register_buffer("perm", SignedPermutation(perm).perm)

    def _symmetrize(self, std: torch.Tensor) -> torch.Tensor:
        """Average the standard deviation over the mirrored output pairs."""
        return 0.5 * (std + std.index_select(-1, self.perm))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian, with the standard deviation averaged over mirrored pairs."""
        mean = mlp_output
        if self.std_type == "scalar":
            std = self._symmetrize(self.std_param).clamp(self.std_range[0], self.std_range[1])
        elif self.std_type == "log":
            log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
            std = self._symmetrize(torch.exp(log_std))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.std_type}. Should be 'scalar' or 'log'.")
        self._distribution = Normal(mean, std)
