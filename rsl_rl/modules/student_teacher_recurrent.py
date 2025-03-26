# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class StudentTeacherRecurrent(StudentTeacher):
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,
        teacher_recurrent=False,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "StudentTeacherRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str(kwargs.keys()),
            )

        self.teacher_recurrent = teacher_recurrent

        super().__init__(
            num_student_obs=rnn_hidden_dim,
            num_teacher_obs=rnn_hidden_dim if teacher_recurrent else num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        self.memory_s = Memory(num_student_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        if self.teacher_recurrent:
            self.memory_t = Memory(
                num_teacher_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
            )

        print(f"Student RNN: {self.memory_s}")
        if self.teacher_recurrent:
            print(f"Teacher RNN: {self.memory_t}")

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.memory_s.reset(dones, hidden_states[0])
        if self.teacher_recurrent:
            self.memory_t.reset(dones, hidden_states[1])

    def act(self, observations):
        input_s = self.memory_s(observations)
        return super().act(input_s.squeeze(0))

    def act_inference(self, observations):
        input_s = self.memory_s(observations)
        return super().act_inference(input_s.squeeze(0))

    def evaluate(self, teacher_observations):
        if self.teacher_recurrent:
            teacher_observations = self.memory_t(teacher_observations)
        return super().evaluate(teacher_observations.squeeze(0))

    def get_hidden_states(self):
        if self.teacher_recurrent:
            return self.memory_s.hidden_states, self.memory_t.hidden_states
        else:
            return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        self.memory_s.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)
