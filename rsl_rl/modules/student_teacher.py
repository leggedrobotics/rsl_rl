# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState


class StudentTeacher(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False  # Indicates if teacher has been loaded

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "The StudentTeacher module only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        # Student
        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")

        # Student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Teacher
        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        print(f"Teacher MLP: {self.teacher}")

        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        # Compute mean
        mean = self.student(obs)
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # Make sure teacher is in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters.
        """
        # Check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict):  # Load parameters from rl training
            # Rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False  # Training does not resume
        elif any("student" in key for key in state_dict):  # Load parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # Set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True  # Training resumes
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
