# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import FlashSAC
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


class OffPolicyRunner(OnPolicyRunner):
    """Off-policy runner for replay-based algorithms (e.g. FlashSAC).

    Reuses :class:`~rsl_rl.runners.OnPolicyRunner` for construction, logging,
    checkpointing, and policy export, and overrides :meth:`learn` with an
    off-policy loop: each iteration collects ``num_steps_per_env`` environment
    steps into the algorithm's replay buffer, then performs
    ``num_steps_per_env * updates_per_step`` gradient updates.

    Multi-GPU training is not supported; it raises during construction.
    """

    alg: FlashSAC
    """The off-policy algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner and reject multi-GPU configurations."""
        if int(os.getenv("WORLD_SIZE", "1")) > 1:
            raise NotImplementedError(
                "OffPolicyRunner does not support multi-GPU training (WORLD_SIZE > 1) in this version."
            )
        super().__init__(env, train_cfg, log_dir, device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the off-policy learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration).
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        num_steps_per_env = self.cfg["num_steps_per_env"]
        updates_per_step = self.cfg["updates_per_step"]

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()

            # Collect environment interactions into the replay buffer.
            with torch.inference_mode():
                for _ in range(num_steps_per_env):
                    actions = self.alg.act(obs, training=True)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)

            collect_time = time.time() - start
            start = time.time()

            # Gradient phase: several optimization steps over sampled replay batches.
            num_grad_steps = round(num_steps_per_env * updates_per_step)
            # Aggregate metrics across the iteration's updates by averaging each
            # key over the updates that reported it. This ensures metrics that
            # only appear on delayed-actor steps (actor/entropy/temperature) are
            # still surfaced, rather than being dropped by keeping only the last
            # update's dict.
            loss_sums: dict[str, float] = {}
            loss_counts: dict[str, int] = {}
            for _ in range(num_grad_steps):
                for key, value in self.alg.update().items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + value
                    loss_counts[key] = loss_counts.get(key, 0) + 1
            loss_dict = {key: loss_sums[key] / loss_counts[key] for key in loss_sums}

            learn_time = time.time() - start
            self.current_learning_iteration = it

            # Log information.
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=loss_dict.get("learning_rate", 0.0),
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            # Save model.
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer.
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
