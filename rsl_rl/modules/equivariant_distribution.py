# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from torch.distributions import Normal
from typing import Any

from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.modules.equivariant import SignedPermutation

__all__ = ["EquivariantGaussianDistribution"]


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
            **kwargs: Forwarded to :class:`~rsl_rl.modules.GaussianDistribution`.
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
