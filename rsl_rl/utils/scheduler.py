# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code ported from FlashSAC (flash_rl):
#   Copyright (c) 2026 Holiday Robotics
#   SPDX-License-Identifier: MIT
# ``warmup_cosine_decay_scheduler`` is adapted from
# ``flash_rl/agents/utils/scheduler.py``.

"""Learning-rate schedules for off-policy algorithms."""

from __future__ import annotations

import math
from typing import Callable


def warmup_cosine_decay_scheduler(
    init_value: float,
    peak_value: float,
    end_value: float,
    warmup_steps: int,
    decay_steps: int,
) -> Callable[[int], float]:
    """Return a warmup-then-cosine-decay learning-rate schedule.

    Linearly warms up from ``init_value`` to ``peak_value`` over ``warmup_steps``,
    then cosine-decays from ``peak_value`` to ``end_value`` until ``decay_steps``
    (optax convention: ``decay_steps`` is the total schedule length), then holds
    at ``end_value``.

    Args:
        init_value: Learning rate at step 0.
        peak_value: Learning rate at the end of warmup.
        end_value: Learning rate after decay completes.
        warmup_steps: Number of linear warmup steps.
        decay_steps: Total schedule length (warmup + decay).

    Returns:
        A function mapping an integer step to a learning-rate multiplier value.
    """

    def scheduler(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        if step < decay_steps:
            decay_step = step - warmup_steps
            progress = decay_step / max(1, (decay_steps - warmup_steps))
            return end_value + (peak_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))
        return end_value

    return scheduler
