# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Storage for the learning algorithms."""

from .rollout_storage import RolloutStorage
from .multi_critic_rollout_storage import MultiCriticRolloutStorage

__all__ = ["RolloutStorage"]
