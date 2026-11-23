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
    """Muon for 2D weight matrices, with update magnitudes matched to Adam and CUDA graph support.

    The momentum is orthogonalized with :func:`orthogonalize` and scaled by ``0.2 * sqrt(max(rows, cols))`` so the
    update RMS matches Adam's, which lets Muon reuse Adam learning rates. The learning rate may be a tensor; float
    learning rates are rounded to float32 tensors as well, so CUDA graphs with tensor learning rates reproduce eager
    updates exactly. Setting ``found_inf`` to a tensor that is nonzero skips the step on the device, like fused Adam.

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
        """Initialize the optimizer; every parameter must be a 2D matrix."""
        if not 0 <= momentum < 1 or weight_decay < 0 or ns_steps < 1:
            raise ValueError("Invalid Muon momentum, weight decay or number of Newton-Schulz steps.")
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.ns_steps = ns_steps
        self.found_inf: torch.Tensor | None = None
        for group in self.param_groups:
            if any(param.ndim != 2 for param in group["params"]):
                raise ValueError("Muon only optimizes 2D weight matrices.")

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Apply one update, leaving parameters and momentum unchanged when ``found_inf`` is set."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        skip = None if self.found_inf is None else self.found_inf.bool()
        for group in self.param_groups:
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
