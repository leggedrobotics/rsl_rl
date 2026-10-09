# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code ported from FlashSAC (flash_rl):
#   Copyright (c) 2026 Holiday Robotics
#   SPDX-License-Identifier: MIT
# The FlashSAC layers, distributional value head, squashed-Gaussian policy, and
# weight-normalization ("Unit*") modules below are adapted from
# ``flash_rl/agents/flashSAC/layer.py``, ``.../network.py`` and
# ``.../utils/distribution.py``. Adapted to live inside rsl_rl as reusable
# building blocks for the SAC algorithm.

"""FlashSAC neural-network building blocks.

These are self-contained ``nn.Module`` s that provide three capabilities RSL-RL's
default :class:`~rsl_rl.modules.MLP` / :class:`~rsl_rl.modules.Distribution` do
not offer, and which the FlashSAC algorithm relies on:

1. **Weight normalization.** Every learnable layer implements
   :meth:`normalize_parameters`, which the SAC algorithm calls after each
   optimizer step to renormalize weights onto the unit sphere.
2. **A squashed (tanh) Gaussian policy** with a Jacobian-corrected log-prob
   (:class:`NormalTanhPolicy`).
3. **An ensembled categorical (distributional / C51-style) value head**
   (:class:`EnsembleCategoricalValue`) for clipped double-Q learning.

They are kept separate from the PPO ``MLP`` so the two algorithms do not share
mutable state.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Return a numerically stable ``log|det d tanh(x)/dx|``.

    Uses ``log(1 - tanh^2(x)) = 2 * (log 2 - x - softplus(-2x))`` to avoid the
    catastrophic cancellation of a naive ``log(1 - tanh(x)**2)``.
    """
    return 2.0 * (math.log(2.0) - x - nn.functional.softplus(-2.0 * x))


# -------------------------------------------------------------------------
# Single-network weight-normalized layers.
# -------------------------------------------------------------------------


class UnitLinear(nn.Module):
    """Bias-free linear layer whose output rows are renormalized to unit norm."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the linear layer with orthogonal weights."""
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear map."""
        return self.w(x)  # type: ignore[no-any-return]

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Normalize each output feature's weight vector to unit L2 norm.

        Normalizes along ``dim=-1`` (input features) so each output feature is a
        unit vector. Called after initialization and after each optimizer step.
        """
        self.w.weight.copy_(nn.functional.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """BatchNorm whose affine parameters are renormalized to ``sqrt(d)``."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        """Initialize affine parameters and running statistics."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply batch normalization (uses running stats when ``training`` is False)."""
        return nn.functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Normalize the concatenated (scale, bias) vector to norm ``sqrt(d)``."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    """RMSNorm whose scale is renormalized to norm ``sqrt(d)``."""

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        """Initialize the RMSNorm scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization over the last dimension."""
        # Manual RMS norm (equivalent to F.rms_norm over the last dim) so the
        # module is ONNX-exportable; F.rms_norm has no ONNX symbolic at opset 18.
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Normalize the scale vector to norm ``sqrt(d)``."""
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    """Input projection: ``UnitBatchNorm`` then ``UnitLinear``."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize the input normalization and projection."""
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Normalize then project the input."""
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class FlashSACBlock(nn.Module):
    """Residual MLP block with expansion and ``UnitBatchNorm``."""

    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        """Initialize the two-layer residual block."""
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply the residual block."""
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = nn.functional.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = nn.functional.relu(x)
        x = x + residual
        return x


class NormalTanhPolicy(nn.Module):
    """Squashed (tanh) diagonal-Gaussian policy head.

    ``forward`` returns ``(tanh_action, {"log_prob": ...})`` where the log-prob
    includes the tanh change-of-variables correction. ``get_mean_and_std``
    exposes the pre-squash Gaussian parameters for deterministic inference.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        """Initialize the mean and (squashed) log-std heads."""
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_mean_and_std(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pre-squash Gaussian ``(mean, std)``."""
        del training  # kept for a uniform call signature
        # Functional linear so the weight-normalized ``UnitLinear`` weights are
        # used directly (also plays nicely with AMP).
        mean = nn.functional.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = nn.functional.linear(x, self.std_w.w.weight, self.std_bias)
        # Squash log-std into [log_std_min, log_std_max] for stability.
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw_log_std))
        std = torch.exp(log_std)
        return mean, std

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample a squashed action and its Jacobian-corrected log-prob."""
        mean, std = self.get_mean_and_std(x, training)
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)
        # Log-prob with tanh Jacobian correction, summed over action dims.
        log_prob = dist.log_prob(raw_action)  # type: ignore[no-untyped-call]
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        log_prob = log_prob.sum(-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return tanh_action, info


# -------------------------------------------------------------------------
# Ensembled layers for the double critic. Internal layout is (N, batch, dim).
# -------------------------------------------------------------------------


class EnsembleUnitLinear(nn.Module):
    """Batched (ensembled) bias-free linear layer with unit-norm rows."""

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int) -> None:
        """Initialize ``num_ensemble`` orthogonal weight matrices."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-ensemble linear map to ``[N, B, in]`` input."""
        # [N, B, in] x [N, out, in] -> [N, B, out]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Normalize each output feature's weight vector to unit L2 norm."""
        self.weight.copy_(nn.functional.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    """Batched (ensembled) BatchNorm operating on ``[N, B, dim]`` input."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        """Initialize per-ensemble affine parameters and running statistics."""
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply per-ensemble batch normalization over the batch dimension."""
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                batch_size = x.shape[1]
                # Cast to float32 for running stats (BatchNorm keeps float32 stats even in AMP).
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (batch_size / (batch_size - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Normalize the concatenated (scale, bias) vector to norm ``sqrt(d)``."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    """Batched (ensembled) RMSNorm operating on ``[N, B, dim]`` input."""

    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6) -> None:
        """Initialize per-ensemble RMSNorm scales."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-ensemble RMS normalization over the last dimension."""
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        """Normalize the scale vector to norm ``sqrt(d)``."""
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    """Batched (ensembled) input projection."""

    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int) -> None:
        """Initialize the per-ensemble normalization and projection."""
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Normalize then project the per-ensemble input."""
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleFlashSACBlock(nn.Module):
    """Batched (ensembled) residual MLP block."""

    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4) -> None:
        """Initialize the per-ensemble two-layer residual block."""
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Apply the per-ensemble residual block."""
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = nn.functional.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = nn.functional.relu(x)
        x = x + residual
        return x


class EnsembleCategoricalValue(nn.Module):
    """Ensembled categorical (C51-style) value head.

    Returns ``(value, {"log_prob": ...})`` where ``value`` is the expected value
    over a fixed support ``[min_v, max_v]`` and ``log_prob`` are per-bin
    log-probabilities of shape ``(num_ensemble, batch, num_bins)``.
    """

    bin_values: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ) -> None:
        """Initialize the per-ensemble logits head and the value support."""
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the expected value and per-bin log-probabilities."""
        del training  # kept for a uniform call signature
        value = self.w(x) + self.bias.unsqueeze(1)
        log_prob = nn.functional.log_softmax(value, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return value, info


# -------------------------------------------------------------------------
# Composed networks.
# -------------------------------------------------------------------------


class FlashSACActor(nn.Module):
    """Squashed-Gaussian actor: embedder -> residual blocks -> RMSNorm -> policy."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        """Initialize the actor trunk and policy head."""
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    def _trunk(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """Run the shared trunk (embedder + blocks + post-norm)."""
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return x

    def get_mean_and_std(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pre-squash Gaussian ``(mean, std)`` for the observations."""
        x = self._trunk(observations, training)
        return self.predictor.get_mean_and_std(x, training)

    def forward(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a sampled squashed action and its log-prob."""
        x = self._trunk(observations, training)
        return self.predictor(x, training)


class FlashSACDoubleCritic(nn.Module):
    """Clipped double-Q distributional critic (``num_qs`` fused ensemble members).

    Reference: Fujimoto et al., "Addressing Function Approximation Error in
    Actor-Critic Methods" (https://arxiv.org/pdf/1802.09477v3).
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ) -> None:
        """Initialize the ensembled critic trunk and distributional value head."""
        super().__init__()
        self.num_qs = num_qs
        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return per-ensemble Q-values and per-bin log-probabilities for ``(obs, action)``."""
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACTemperature(nn.Module):
    """Learnable entropy temperature stored in log-space; ``forward()`` returns ``alpha``."""

    def __init__(self, initial_value: float = 0.01) -> None:
        """Initialize the log-temperature parameter."""
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Return the current temperature ``alpha = exp(log_temp)``."""
        return torch.exp(self.log_temp)
