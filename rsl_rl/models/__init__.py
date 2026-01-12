# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .actor_critic import ActorCritic
from .actor_critic_cnn import ActorCriticCNN
from .actor_critic_recurrent import ActorCriticRecurrent
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

__all__ = [
    "ActorCritic",
    "ActorCriticCNN",
    "ActorCriticRecurrent",
    "StudentTeacher",
    "StudentTeacherRecurrent",
]
