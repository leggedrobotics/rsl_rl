# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from rsl_rl.algorithms import SAC
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class OffPolicyRunner(OnPolicyRunner):
    """Off-policy runner for reinforcement learning algorithms.

    This runner uses the interaction, logging, and checkpointing loop provided by :class:`OnPolicyRunner`.
    """

    alg: SAC
    """The SAC algorithm."""
