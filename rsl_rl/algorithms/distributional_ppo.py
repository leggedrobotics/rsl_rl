# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from .ppo import PPO


def quantile_huber_loss(quantiles: torch.Tensor, targets: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Compute Quantile Huber loss between predicted quantiles and scalar targets.

    Reference:
        - Dabney et al. "Distributional Reinforcement Learning with Quantile Regression." AAAI (2018).

    Args:
        quantiles: Predicted quantiles tensor of shape (..., num_quantiles).
        targets: Target values tensor of shape (..., 1) or (...).
        kappa: Huber loss threshold parameter (default: 1.0).

    Returns:
        Scalar quantile Huber loss tensor.
    """
    if targets.dim() == 1 or targets.shape[-1] != 1:
        targets = targets.unsqueeze(-1)

    num_quantiles = quantiles.shape[-1]
    tau = (torch.arange(num_quantiles, device=quantiles.device, dtype=quantiles.dtype) + 0.5) / num_quantiles
    tau = tau.unsqueeze(0)

    pairwise_errors = targets - quantiles
    abs_errors = torch.abs(pairwise_errors)

    huber_loss = torch.where(
        abs_errors <= kappa,
        0.5 * pairwise_errors**2,
        kappa * (abs_errors - 0.5 * kappa),
    )

    check_weight = torch.abs(tau - (pairwise_errors < 0).to(quantiles.dtype))
    loss = check_weight * huber_loss

    return torch.mean(loss)


class DistributionalPPO(PPO):
    """Distributional PPO using Quantile Regression and Huber Loss.

    Extends standard PPO by estimating return distributions via quantiles
    rather than a single scalar expected value. Scalar value estimates for GAE
    are obtained by reducing over quantiles.

    Reference:
        - Dabney et al. "Distributional Reinforcement Learning with Quantile Regression." AAAI (2018).
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_quantiles: int = 8,
        quantile_huber_kappa: float = 1.0,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
    ) -> None:
        """Initialize DistributionalPPO with quantile regression settings."""
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            use_mixed_precision=use_mixed_precision,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
            grad_reduce_bucket_mb=grad_reduce_bucket_mb,
        )
        self.num_quantiles = num_quantiles
        self.quantile_huber_kappa = quantile_huber_kappa

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DistributionalPPO:
        """Construct the DistributionalPPO algorithm with quantile critic output dimension."""
        cfg["critic"].setdefault("output_dim", cfg["algorithm"].get("num_quantiles", 8))
        return PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[return-value]

    def compute_value_estimate(self, obs: TensorDict) -> torch.Tensor:
        """Compute mean scalar value estimate from quantiles for GAE advantage estimation."""
        critic_output = super().compute_value_estimate(obs)
        if self.num_quantiles > 0:
            return critic_output.mean(dim=-1, keepdim=True)
        return critic_output

    def compute_critic_loss(
        self,
        value_preds: torch.Tensor,
        returns: torch.Tensor,
        old_value_preds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Quantile Huber loss over predicted quantiles."""
        if self.num_quantiles > 0:
            return quantile_huber_loss(value_preds, returns, kappa=self.quantile_huber_kappa)
        return super().compute_critic_loss(value_preds, returns, old_value_preds)
