# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner for MultiCriticPPO with multiple reward streams."""

from __future__ import annotations

import os
import time

import torch

from rsl_rl.algorithms import MultiCriticPPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan


class MultiCriticOnPolicyRunner(OnPolicyRunner):
    """On-policy runner specialized for MultiCriticPPO.

    Unlike :class:`OnPolicyRunner`, this runner expects the environment to
    return one reward tensor per critic:

        obs, (reward_0, reward_1, ...), dones, extras

    The base OnPolicyRunner remains unchanged and continues to support
    standard PPO algorithms with a single reward tensor.
    """

    alg: MultiCriticPPO

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Construct a MultiCriticPPO runner."""
        super().__init__(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
        )

        if not isinstance(self.alg, MultiCriticPPO):
            raise TypeError(
                "MultiCriticOnPolicyRunner requires MultiCriticPPO, "
                f"but got {type(self.alg).__name__}."
            )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> list[dict[str, float]]:
        """Run the MultiCriticPPO learning loop."""

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(
                f"Synchronizing parameters for rank "
                f"{self.gpu_global_rank}..."
            )
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        loss_history: list[dict[str, float]] = []

        for it in range(start_it, total_it):
            start = time.time()

            # ---------------------------------------------------------
            # Rollout
            # ---------------------------------------------------------
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Actor produces one action tensor.
                    actions = self.alg.act(obs)

                    # MultiCriticPPO environments return one reward
                    # tensor per critic.
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        for reward in rewards:
                            check_nan(obs, reward, dones)

                    # Move to device
                    obs = obs.to(self.device)
                    rewards = tuple(reward.to(self.device) for reward in rewards)
                    dones = dones.to(self.device)

                    # Process multi-critic step
                    self.alg.process_env_step(
                        obs,
                        rewards,
                        dones,
                        extras,
                    )

                    # Intrinsic rewards, if applicable
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )

                    logging_rewards = torch.stack(rewards, dim=0).mean(dim=0)

                    # Logging
                    self.logger.process_env_step(
                        logging_rewards,
                        dones,
                        extras,
                        intrinsic_rewards,
                    )

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns for every critic.
                self.alg.compute_returns(obs)

            # ---------------------------------------------------------
            # Update
            # ---------------------------------------------------------
            loss_dict = self.alg.update()

            loss_history.append(
                {
                    key: value.detach().item()
                    if isinstance(value, torch.Tensor)
                    else float(value)
                    for key, value in loss_dict.items()
                }
            )

            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it

            # ---------------------------------------------------------
            # Logging
            # ---------------------------------------------------------
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=(
                    self.alg.rnd.weight
                    if self.cfg["algorithm"].get("rnd_cfg")
                    else None
                ),
            )

            # ---------------------------------------------------------
            # Save checkpoint
            # ---------------------------------------------------------
            if (
                self.logger.writer is not None
                and self.logger.log_dir is not None
                and it % self.cfg["save_interval"] == 0
            ):
                self.save(
                    os.path.join(
                        self.logger.log_dir,
                        f"model_{it}.pt",
                    )
                )

        # -------------------------------------------------------------
        # Final checkpoint
        # -------------------------------------------------------------
        if (
            self.logger.writer is not None
            and self.logger.log_dir is not None
        ):
            self.save(
                os.path.join(
                    self.logger.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )
            self.logger.stop_logging_writer()

        return loss_history