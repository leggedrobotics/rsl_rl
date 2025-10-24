# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import neptune
except ModuleNotFoundError:
    raise ModuleNotFoundError("neptune-client is required to log to Neptune.") from None


class NeptuneLogger:
    def __init__(self, project: str, token: str) -> None:
        self.run = neptune.init_run(project=project, api_token=token)

    def store_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        self.run["runner_cfg"] = runner_cfg
        self.run["policy_cfg"] = policy_cfg
        self.run["alg_cfg"] = alg_cfg
        self.run["env_cfg"] = asdict(env_cfg)


class NeptuneSummaryWriter(SummaryWriter):
    """Summary writer for Neptune."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["neptune_project"]
        except KeyError:
            raise KeyError("Please specify neptune_project in the runner config, e.g. legged_gym.") from None

        try:
            token = os.environ["NEPTUNE_API_TOKEN"]
        except KeyError:
            raise KeyError(
                "Neptune api token not found. Please run or add to ~/.bashrc: export NEPTUNE_API_TOKEN=YOUR_API_TOKEN"
            ) from None

        try:
            entity = os.environ["NEPTUNE_USERNAME"]
        except KeyError:
            raise KeyError(
                "Neptune username not found. Please run or add to ~/.bashrc: export NEPTUNE_USERNAME=YOUR_USERNAME"
            ) from None

        neptune_project = entity + "/" + project
        self.neptune_logger = NeptuneLogger(neptune_project, token)
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }
        run_name = os.path.split(log_dir)[-1]
        self.neptune_logger.run["log_dir"].log(run_name)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        self.neptune_logger.run[self._map_path(tag)].log(scalar_value, step=global_step)

    def stop(self) -> None:
        self.neptune_logger.run.stop()

    def log_config(self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict) -> None:
        self.neptune_logger.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path: str, iter: int) -> None:
        self.neptune_logger.run["model/saved_model_" + str(iter)].upload(model_path)

    def save_file(self, path: str) -> None:
        name = path.rsplit("/", 1)[-1].split(".")[0]
        self.neptune_logger.run["git_diff/" + name].upload(path)

    def _map_path(self, path: str) -> str:
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path
