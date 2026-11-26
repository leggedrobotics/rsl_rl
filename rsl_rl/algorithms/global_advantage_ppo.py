# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional PPO variant with pooled rollout advantage normalization."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from .ppo import PPO


class GlobalAdvantagePPO(PPO):
    """Normalize rollout advantages across distributed ranks before PPO updates.

    This opt-in variant treats all ranks as shards of one rollout. It accepts the
    same settings as :class:`PPO`, except that per-mini-batch advantage
    normalization must be disabled. Single-process behavior is unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize PPO and reject incompatible local mini-batch normalization."""
        super().__init__(*args, **kwargs)
        if self.normalize_advantage_per_mini_batch:
            raise ValueError("GlobalAdvantagePPO requires normalize_advantage_per_mini_batch=False.")

    @torch.no_grad()
    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE targets, then normalize with pooled sample statistics."""
        super().compute_returns(obs)
        if not self.is_multi_gpu or self.gpu_world_size == 1:
            return

        with torch.inference_mode(False):
            advantages = (self.storage.returns - self.storage.values).to(torch.float64)
            count_sum = torch.stack((advantages.new_tensor(advantages.numel()), advantages.sum()))
            torch.distributed.all_reduce(count_sum, op=torch.distributed.ReduceOp.SUM)
            if count_sum[0] <= 1:
                return
            centered = advantages - count_sum[1] / count_sum[0]
            squared_deviation = centered.square().sum()
            torch.distributed.all_reduce(squared_deviation, op=torch.distributed.ReduceOp.SUM)
            std = (squared_deviation / (count_sum[0] - 1)).sqrt()
            self.storage.advantages = (centered / (std + 1e-8)).to(self.storage.advantages.dtype)
