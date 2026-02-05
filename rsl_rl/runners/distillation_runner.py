# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.algorithms import Distillation
from rsl_rl.runners import OnPolicyRunner


class DistillationRunner(OnPolicyRunner):
    """Distillation runner for training and evaluation of teacher-student methods."""

    alg: Distillation
    """The distillation algorithm."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Check if teacher is loaded
        if not self.alg.teacher_loaded:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        super().learn(num_learning_iterations, init_at_random_ep_len)
