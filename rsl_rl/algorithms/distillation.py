# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        device="cpu",
    ):
        self.device = device
        self.learning_rate = learning_rate

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=self.learning_rate)
        self.transition = RolloutStorage.Transition()

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def act(self, obs, teacher_obs):
        # compute the actions
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behaviour_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):  # TODO unify num_steps_per_env and gradient_length
            self.policy.reset()
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions in self.storage.generator():

                # inference the student for gradient computation
                actions = self.policy.act_inference(obs)

                # behaviour cloning loss
                behaviour_loss = nn.functional.mse_loss(actions, privileged_actions)

                # total loss
                loss = loss + behaviour_loss

                mean_behaviour_loss += behaviour_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

        mean_behaviour_loss /= cnt
        self.storage.clear()
        self.policy.reset()  # TODO needed?

        # construct the loss dictionary
        loss_dict = {"behaviour": mean_behaviour_loss}

        return loss_dict
