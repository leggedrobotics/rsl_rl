# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # yield the mini-batch
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, rnd_state_batch

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch

                first_traj = last_traj


    # for reinforcement learning with recurrent networks + RL^2 style input (obs + prev_action)
    def recurrent_mini_batch_generator_with_prev_action(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        # pad observations
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        # pad rnd state if exists
        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        # pad actions (for RL^2 input as prev_action)
        padded_action_trajectories, _ = split_and_pad_trajectories(self.actions, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                # 新增 prev_action batch (for RL^2 input)
                prev_actions_batch = padded_action_trajectories[:, first_traj:last_traj]
                # RL^2: shift prev_actions by 1 step
                prev_actions_batch = torch.cat(
                    [torch.zeros_like(prev_actions_batch[:, :1, :], device=prev_actions_batch.device),
                    prev_actions_batch[:, :-1, :]], dim=1
                )

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # hidden states reshape
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                # 新增 prev_actions_batch 输出
                yield obs_batch, prev_actions_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch

                first_traj = last_traj


    # for reinforcement learning with recurrent networks + RL^2 style input (obs + prev_action) + shuffle
    def recurrent_mini_batch_generator_with_prev_action_shuffle(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        # pad observations
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        # pad rnd state if exists
        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        # pad actions (for RL^2 input as prev_action)
        padded_action_trajectories, _ = split_and_pad_trajectories(self.actions, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # shuffle the env
            shuffle_env_index = torch.randperm(self.num_envs, device=self.device)

            # re-pad the trajectories according to the shuffled env index
            padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
                self.observations[:, shuffle_env_index],
                self.dones[:, shuffle_env_index],
            )
            if self.privileged_observations is not None:
                padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(
                    self.privileged_observations[:, shuffle_env_index],
                    self.dones[:, shuffle_env_index],
                )
            else:
                padded_privileged_obs_trajectories = padded_obs_trajectories

            # re-pad rnd state if exists
            if self.rnd_state_shape is not None:
                padded_rnd_state_trajectories, _ = split_and_pad_trajectories(
                    self.rnd_state[:, shuffle_env_index],
                    self.dones[:, shuffle_env_index],
                )
            else:
                padded_rnd_state_trajectories = None

            # re-pad actions (for RL^2 input as prev_action)
            padded_action_trajectories, _ = split_and_pad_trajectories(
                self.actions[:, shuffle_env_index],
                self.dones[:, shuffle_env_index],
            )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                shuffled_batch_envs = shuffle_env_index[start:stop]

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, shuffled_batch_envs])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                # 新增 prev_action batch (for RL^2 input)
                prev_actions_batch = padded_action_trajectories[:, first_traj:last_traj]
                # RL^2: shift prev_actions by 1 step
                prev_actions_batch = torch.cat(
                    [torch.zeros_like(prev_actions_batch[:, :1, :], device=prev_actions_batch.device),
                    prev_actions_batch[:, :-1, :]], dim=1
                )

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, shuffled_batch_envs]
                old_mu_batch = self.mu[:, shuffled_batch_envs]
                old_sigma_batch = self.sigma[:, shuffled_batch_envs]
                returns_batch = self.returns[:, shuffled_batch_envs]
                advantages_batch = self.advantages[:, shuffled_batch_envs]
                values_batch = self.values[:, shuffled_batch_envs]
                old_actions_log_prob_batch = self.actions_log_prob[:, shuffled_batch_envs]

                # hidden states reshape
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                # 新增 prev_actions_batch 输出
                yield obs_batch, prev_actions_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch

                first_traj = last_traj


    # chunk 生成器 9.16
    def debug_chunk_mini_batch_generator(self, num_mini_batches, num_epochs=8, chunk_size=1):
        """
        Generator that yields mini-batches of experience in chunks of length `chunk_size`,

        Yields:
            Tuple: (
                obs, privileged_obs, actions, values, returns,
                log_probs, advantages, old_mu, old_sigma,
                hid_states(first), masks(None), rnd_state
            )
            Shapes:
                obs: [chunk_size, batch, obs_dim]
                masks: [chunk_size, batch, 1] (1=valid, 0=padding)
                hidden_state: [num_layers, batch, hidden_dim]
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        T = self.num_transitions_per_env
        N = self.num_envs
        assert T % chunk_size == 0, "T must be divisible by chunk_size"

        num_chunks = T // chunk_size
        batch_size = num_chunks * N
        mini_batch_size = batch_size // num_mini_batches

        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)

        # 切分chunk函数
        def make_chunks(x):
            T, N = x.shape[:2]
            num_chunks = T // chunk_size

            x = x.view(num_chunks, chunk_size, N, *x.shape[2:])
            x = x.transpose(0, 1)

            x = x.reshape(chunk_size, num_chunks * N, *x.shape[3:])
            return x

        # return hidden_chunks_first  # [num_layers, batch, hidden_dim]
        def get_chunk_hidden_states(saved_hidden_states):
            """
            saved_hidden_states: list of [T, num_layers, num_envs, hidden_dim]
            returns: tensor [num_layers, total_chunks, hidden_dim]
            """
            if len(saved_hidden_states) == 1:
                h = saved_hidden_states[0]  # [T, num_layers, num_envs, hidden_dim]
            else:
                h = torch.stack(saved_hidden_states, dim=1)  # [T, num_layers, num_envs, hidden_dim] or list stacking

            # take first timestep of each chunk
            h = h.permute(2, 0, 1, 3)  # [num_envs, T, num_layers, hidden_dim]
            h_chunks = h.view(N, num_chunks, chunk_size, h.shape[2], h.shape[3])
            h_first = h_chunks[:, :, 0, :, :]  # [num_envs, num_chunks, num_layers, hidden_dim]
            h_first = h_first.permute(2, 0, 1, 3).reshape(h_first.shape[2], num_chunks*N, h_first.shape[3])
            return h_first  # [num_layers, batch, hidden_dim]

        # Core
        observations = make_chunks(self.observations)
        if self.privileged_observations is not None:
            privileged_observations = make_chunks(self.privileged_observations)
        else:
            privileged_observations = observations

        actions = make_chunks(self.actions)
        values = make_chunks(self.values)
        returns = make_chunks(self.returns)

        # prev_actions for RL^2 input
        prev_actions = torch.zeros_like(self.actions, device=self.actions.device)
        prev_actions[1:, :, :] = self.actions[:-1, :, :]
        prev_actions = make_chunks(prev_actions)

        # For PPO
        old_actions_log_prob = make_chunks(self.actions_log_prob)
        advantages = make_chunks(self.advantages)
        old_mu = make_chunks(self.mu)
        old_sigma = make_chunks(self.sigma)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = make_chunks(self.rnd_state)

        # For hidden_states_first
        hid_a_chunks_first = get_chunk_hidden_states(self.saved_hidden_states_a)
        hid_c_chunks_first = get_chunk_hidden_states(self.saved_hidden_states_c)

        # 为了对齐后面RNN的输入，直接生成一个全true的mask
        # 必须mask有值才会用输入的hid，mask=None默认用memory自己存的hid
        masks_batch = torch.ones((chunk_size, mini_batch_size), dtype=torch.bool, device=observations.device)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # [batch_size, batch, dim]，记得切第二维
                # -- Core
                obs_batch = observations[:, batch_idx]
                privileged_observations_batch = privileged_observations[:, batch_idx]
                actions_batch = actions[:, batch_idx]
                prev_actions_batch = prev_actions[:, batch_idx]

                # -- For PPO
                target_values_batch = values[:, batch_idx]
                returns_batch = returns[:, batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[:, batch_idx]
                advantages_batch = advantages[:, batch_idx]
                old_mu_batch = old_mu[:, batch_idx]
                old_sigma_batch = old_sigma[:, batch_idx]

                # hidden_state_first
                hid_a_batch = hid_a_chunks_first[:, batch_idx]
                hid_c_batch = hid_c_chunks_first[:, batch_idx]
                hid_states_batch = (hid_a_batch, hid_c_batch)

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[:, batch_idx]
                else:
                    rnd_state_batch = None

                # yield the mini-batch
                yield obs_batch, prev_actions_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch


    # 旧的chunk生成器 已经弃用
    def chunk_mini_batch_generator(self, num_mini_batches, num_epochs=8, chunk_size=2):
        """
        Fully vectorized chunk mini-batch generator.
        Outputs are identical to your previous generator but with much less Python overhead.

        Shapes:
            obs: [chunk_size, batch, obs_dim]
            masks: [chunk_size, batch]
            hidden_state: [num_layers, batch, hidden_dim]
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        num_envs = self.num_envs
        T = self.num_transitions_per_env
        num_chunks_per_env = T // chunk_size
        total_chunks = num_envs * num_chunks_per_env

        device = self.device

        # ========== 1. Permute to [num_envs, T, dim] ==========
        obs_env = self.observations.permute(1, 0, 2)  # [num_envs, T, obs_dim]
        priv_obs_env = obs_env if self.privileged_observations is None else self.privileged_observations.permute(1, 0, 2)
        actions_env = self.actions.permute(1, 0, 2)
        values_env = self.values.permute(1, 0, 2)
        returns_env = self.returns.permute(1, 0, 2)
        log_probs_env = self.actions_log_prob.permute(1, 0, 2)
        advantages_env = self.advantages.permute(1, 0, 2)
        old_mu_env = self.mu.permute(1, 0, 2)
        old_sigma_env = self.sigma.permute(1, 0, 2)
        dones_env = self.dones.permute(1, 0, 2)

        prev_actions_env = torch.zeros_like(actions_env)
        prev_actions_env[:, 1:, :] = actions_env[:, :-1, :]

        rnd_env = None if self.rnd_state_shape is None else self.rnd_state.permute(1, 0, 2)

        # ========== 2. Create chunk indices ==========
        # chunk shape: [num_envs, num_chunks_per_env, chunk_size, dim]
        def make_chunks(x):
            x = x.view(num_envs, num_chunks_per_env, chunk_size, -1)
            return x

        obs_chunks = make_chunks(obs_env)
        priv_obs_chunks = make_chunks(priv_obs_env)
        actions_chunks = make_chunks(actions_env)
        prev_actions_chunks = make_chunks(prev_actions_env)
        values_chunks = make_chunks(values_env)
        returns_chunks = make_chunks(returns_env)
        log_probs_chunks = make_chunks(log_probs_env)
        advantages_chunks = make_chunks(advantages_env)
        old_mu_chunks = make_chunks(old_mu_env)
        old_sigma_chunks = make_chunks(old_sigma_env)
        masks_chunks = (~make_chunks(dones_env).bool().squeeze(-1))  # [num_envs, num_chunks, chunk_size]

        if rnd_env is not None:
            rnd_chunks = make_chunks(rnd_env)

        # ========== 3. Prepare hidden states (first timestep of each chunk) ==========
        def get_chunk_hidden_states(saved_hidden_states):
            """
            saved_hidden_states: list of [T, num_layers, num_envs, hidden_dim]
            returns: tensor [num_layers, total_chunks, hidden_dim]
            """
            if len(saved_hidden_states) == 1:
                h = saved_hidden_states[0]  # [T, num_layers, num_envs, hidden_dim]
            else:
                h = torch.stack(saved_hidden_states, dim=1)  # [T, num_layers, num_envs, hidden_dim] or list stacking

            # take first timestep of each chunk
            h = h.permute(2, 0, 1, 3)  # [num_envs, T, num_layers, hidden_dim]
            h_chunks = h.view(num_envs, num_chunks_per_env, chunk_size, h.shape[2], h.shape[3])
            h_first = h_chunks[:, :, 0, :, :]  # [num_envs, num_chunks, num_layers, hidden_dim]
            h_first = h_first.permute(2, 0, 1, 3).reshape(h_first.shape[2], total_chunks, h_first.shape[3])
            return h_first  # [num_layers, total_chunks, hidden_dim]

        hid_a_chunks = get_chunk_hidden_states(self.saved_hidden_states_a)
        hid_c_chunks = get_chunk_hidden_states(self.saved_hidden_states_c)
        hid_states_chunks = (hid_a_chunks, hid_c_chunks)

        # ========== 4. Flatten chunks ==========
        def flatten_chunks(x):
            return x.reshape(total_chunks, chunk_size, *x.shape[3:])  # [total_chunks, chunk_size, dim]

        obs_chunks = flatten_chunks(obs_chunks)
        priv_obs_chunks = flatten_chunks(priv_obs_chunks)
        actions_chunks = flatten_chunks(actions_chunks)
        prev_actions_chunks = flatten_chunks(prev_actions_chunks)
        values_chunks = flatten_chunks(values_chunks)
        returns_chunks = flatten_chunks(returns_chunks)
        log_probs_chunks = flatten_chunks(log_probs_chunks)
        advantages_chunks = flatten_chunks(advantages_chunks)
        old_mu_chunks = flatten_chunks(old_mu_chunks)
        old_sigma_chunks = flatten_chunks(old_sigma_chunks)
        masks_chunks = flatten_chunks(masks_chunks)  # [total_chunks, chunk_size]
        if rnd_env is not None:
            rnd_chunks = flatten_chunks(rnd_chunks)

        # ========== 5. Shuffle chunks ==========
        indices = torch.randperm(total_chunks, device=device)

        mini_batch_size = total_chunks // num_mini_batches

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                batch_idx = indices[i * mini_batch_size: (i + 1) * mini_batch_size]

                # index all tensors
                obs_batch = obs_chunks[batch_idx].transpose(0, 1)
                prev_actions_batch = prev_actions_chunks[batch_idx].transpose(0, 1)
                priv_obs_batch = priv_obs_chunks[batch_idx].transpose(0, 1)
                actions_batch = actions_chunks[batch_idx].transpose(0, 1)
                values_batch = values_chunks[batch_idx].transpose(0, 1)
                returns_batch = returns_chunks[batch_idx].transpose(0, 1)
                log_probs_batch = log_probs_chunks[batch_idx].transpose(0, 1)
                advantages_batch = advantages_chunks[batch_idx].transpose(0, 1)
                old_mu_batch = old_mu_chunks[batch_idx].transpose(0, 1)
                old_sigma_batch = old_sigma_chunks[batch_idx].transpose(0, 1)
                masks_batch = masks_chunks[batch_idx].transpose(0, 1)
                hid_a_batch = hid_a_chunks[:, batch_idx, :]
                hid_c_batch = hid_c_chunks[:, batch_idx, :]
                hid_states_batch = (hid_a_batch, hid_c_batch)
                rnd_state_batch = None if rnd_env is None else rnd_chunks[batch_idx].transpose(0, 1)

                yield (
                    obs_batch, prev_actions_batch, priv_obs_batch, actions_batch,
                    values_batch, advantages_batch, returns_batch,
                    log_probs_batch, old_mu_batch, old_sigma_batch,
                    hid_states_batch, masks_batch, rnd_state_batch
                )