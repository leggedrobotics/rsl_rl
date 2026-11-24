# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Callable, Iterable


def orthogonalize(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalize a matrix with the quintic Newton-Schulz iteration in BF16.

    The singular values of the result lie roughly in [0.5, 1.5], as in the reference Muon implementation.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.to(torch.bfloat16)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Optimize 2D weight matrices with Muon updates scaled to match Adam.

    Reference:
        Jordan et al. "Muon: An optimizer for hidden layers in neural networks." (2024).
        Liu et al. "Muon is Scalable for LLM Training." arXiv:2502.16982 (2025).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float | torch.Tensor = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        """Initialize Muon for matrix parameters with a scalar or device tensor learning rate."""
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.ns_steps = ns_steps
        self.found_inf: torch.Tensor | None = None
        for group in self.param_groups:
            if any(param.ndim != 2 for param in group["params"]):
                raise ValueError("Muon only optimizes 2D weight matrices.")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply one update, scaling orthogonalized momentum by ``0.2 * sqrt(max(rows, cols))``.

        Rates use FP32 in eager and CUDA graphs. A nonzero ``found_inf`` skips the update on the device.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        skip = None if self.found_inf is None else self.found_inf.bool()
        for group in self.param_groups:
            if not group["params"]:
                continue
            momentum = group["momentum"]
            lr = torch.as_tensor(group["lr"], dtype=torch.float32, device=group["params"][0].device)
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buffer = state["momentum_buffer"]
                grad = param.grad if skip is None else torch.where(skip, 0.0, param.grad)
                updated = buffer.lerp(grad, 1 - momentum)
                buffer.copy_(updated if skip is None else torch.where(skip, buffer, updated))
                direction = grad.lerp(buffer, momentum) if group["nesterov"] else buffer
                scale = lr * (0.2 * math.sqrt(max(param.shape)))
                if skip is not None:
                    scale = scale * (~skip)
                update = orthogonalize(direction, self.ns_steps).to(param.dtype)
                if group["weight_decay"]:
                    param.mul_(1 - lr * group["weight_decay"] * (1 if skip is None else ~skip))
                param.sub_(scale * update)
        return loss
