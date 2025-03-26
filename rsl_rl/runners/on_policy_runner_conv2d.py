# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticConv2d, EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state


class OnPolicyRunnerConv2d(OnPolicyRunner):
    """Custom on-policy runner for training and evaluation with convolutional actor-critic."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()

        # num_obs = obs.shape[1]
        # if "critic" in extras["observations"]:
        #     num_critic_obs = extras["observations"]["critic"].shape[1]
        # else:
        #     num_critic_obs = num_obs
        num_obs = obs["proprioception"].shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        # Convert from [N, H, W, C] to [C, H, W]
        input_image_shape = obs["vision"].permute(0, 3, 1, 2).shape[1:]
        num_image_obs = torch.prod(torch.tensor(input_image_shape)).item()

        # init the actor-critic networks
        actor_critic: ActorCriticConv2d = ActorCriticConv2d(
            num_obs, num_critic_obs, self.env.num_actions, input_image_shape, **self.policy_cfg
        ).to(self.device)

        # init the ppo algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs + num_image_obs],
            [num_critic_obs + num_image_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs["proprioception"])
        image_obs = obs["vision"].permute(0, 3, 1, 2).flatten(start_dim=1)
        proprioception_obs = obs["proprioception"]
        obs = torch.cat([proprioception_obs, image_obs], dim=1)
        critic_obs = torch.cat([critic_obs, image_obs], dim=1)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions from policy
                    actions = self.alg.act(obs, critic_obs)
                    # Step environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs_proprioceptive = obs["proprioception"]
                    
                    # Move to the agent device
                    obs, rewards, dones = obs_proprioceptive.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Normalize observations
                    obs = self.obs_normalizer(obs)
                    # Extract critic observations and normalize
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"].to(self.device))
                    else:
                        critic_obs = obs

                    # Concatenate image observations with proprioceptive observations
                    
                    obs = torch.cat([obs, image_obs], dim=1)
                    critic_obs = torch.cat([critic_obs, image_obs], dim=1)

                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, dones, infos)

                    # Intrinsic rewards (extracted here only for logging)!
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            # Update policy
            # Note: we keep arguments here since locals() loads them
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    if not os.path.exists(os.path.join(self.log_dir, "models")):
                        os.makedirs(os.path.join(self.log_dir, "models"))
                    self.save(os.path.join(self.log_dir,"models", f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            if not os.path.exists(os.path.join(self.log_dir, "models")):
                os.makedirs(os.path.join(self.log_dir, "models"))
            self.save(os.path.join(self.log_dir, "models", f"model_{self.current_learning_iteration}.pt"))


