# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from rsl_rl.algorithms import SAC
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class OffPolicyRunner(OnPolicyRunner):
    """Named runner for replay algorithms using the standard interaction and logging loop."""

    alg: SAC
