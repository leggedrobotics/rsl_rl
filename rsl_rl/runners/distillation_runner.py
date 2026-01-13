# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable


class DistillationRunner(OnPolicyRunner):
    """Distillation runner for training and evaluation of teacher-student methods."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Check if teacher is loaded
        if not self.alg.teacher_loaded:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        super().learn(num_learning_iterations, init_at_random_ep_len)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load models from previous distillation training
        if any("student_state_dict" in key for key in loaded_dict):
            super().load(path, load_optimizer, map_location)
        # Load models from previous RL training
        elif any("actor_state_dict" in key for key in loaded_dict):
            self.alg.teacher.load_state_dict(loaded_dict["actor_state_dict"])  # Previous actor becomes teacher
            self.current_learning_iteration = 0  # This is not a continuation of training
        self.alg.teacher_loaded = True  # Indicate that the teacher has been loaded successfully
        return loaded_dict["infos"]

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.alg.student

    def get_models(self) -> dict[str, MLPModel]:
        """Return a dict of {name: model} for saving/loading."""
        return {
            "student": self.alg.student,
            "teacher": self.alg.teacher,
        }

    def train_mode(self) -> None:
        self.alg.student.train()
        # Teacher is always in eval mode
        self.alg.teacher.eval()

    def _get_default_obs_sets(self) -> list[str]:
        """Get the the default observation sets required for the algorithm.

        .. note::
            See :func:`resolve_obs_groups` for more details on the handling of observation sets.
        """
        return ["student", "teacher"]

    def _construct_algorithm(self, obs: TensorDict) -> Distillation:
        """Construct the distillation algorithm."""
        # Distillation is not compatible with RND and symmetry extensions
        if self.alg_cfg.get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with Distillation.")
        self.alg_cfg["rnd_cfg"] = None
        if self.alg_cfg.get("symmetry_cfg") is not None:
            raise ValueError("The symmetry extension is not compatible with Distillation.")
        self.alg_cfg["symmetry_cfg"] = None

        # Initialize the policy
        student_class = resolve_callable(self.alg_cfg["student"].pop("student_class_name"))
        student: MLPModel = student_class(
            obs, self.cfg["obs_groups"], "student", self.env.num_actions, **self.alg_cfg["student"]
        ).to(self.device)
        teacher_class = resolve_callable(self.alg_cfg["teacher"].pop("teacher_class_name"))
        teacher: MLPModel = teacher_class(
            obs, self.cfg["obs_groups"], "teacher", self.env.num_actions, **self.alg_cfg["teacher"]
        ).to(self.device)

        # Initialize the storage
        storage = RolloutStorage(
            "distillation", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        # Initialize the algorithm
        alg_class = resolve_callable(self.alg_cfg.pop("class_name"))
        alg: Distillation = alg_class(
            student, teacher, storage, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Set RND configuration to None as it does not apply to distillation
        self.cfg["algorithm"]["rnd_cfg"] = None

        return alg
