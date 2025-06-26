# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pathlib
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )

        # Initialize wandb
        self.shared = cfg.get("shared", False)
        settings = wandb.Settings(x_label="main", mode="shared", x_primary=True) if self.shared else None
        tags = ["log_videos_async"] if self.shared else []
        self.run = wandb.init(project=project, entity=entity, tags=tags, settings=settings)

        # Change generated name to project-number format
        self.run.name = project + self.run.name.split("-")[-1]

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]
        self.run.log({"log_dir": run_name}, step=0 if not self.shared else None)
        self.run.define_metric("*", step_metric="local_step")

        self.saved_videos = {}

    def store_config(self, env_cfg: dict, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict):
        self.run.config.update({"runner_cfg": runner_cfg})
        self.run.config.update({"policy_cfg": policy_cfg})
        self.run.config.update({"alg_cfg": alg_cfg})
        self.run.config.update({"env_cfg": env_cfg})

    def _map_path(self, path: str) -> str:
        return self.name_map.get(path, path)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float | str,
        global_step: int,
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
        self.run.log(
            {self._map_path(tag): scalar_value, "local_step": global_step},
            step=global_step if not self.shared else None,
        )

    def stop(self):
        self.run.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        env_cfg_dict = env_cfg.to_dict()
        self.store_config(env_cfg_dict, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        self.run.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        self.run.save(path, base_path=os.path.dirname(path))

    def log_video_files(self, log_name: str = "Video", video_subdir: str | None = "videos"):
        if video_subdir is not None:
            video_dir = pathlib.Path(os.path.join(self.log_dir, video_subdir))
        else:
            video_dir = pathlib.Path(self.log_dir)
        videos = list(video_dir.rglob("*.mp4"))
        for video in videos:
            video_name = str(video)
            video_size_kb = os.stat(video_name).st_size / 1024
            if video_name not in self.saved_videos.keys():
                self.saved_videos[video_name] = {
                    "size": video_size_kb,
                    "recorded": False,
                    "steps": 0,
                }
            else:
                video_info = self.saved_videos[video_name]
                if video_info["recorded"]:
                    continue
                elif video_info["size"] == video_size_kb and video_size_kb > 100:
                    # wait 10 steps after recording has been completed
                    if video_info["steps"] > 10:
                        self.add_video(video_name, log_name=log_name)
                        self.saved_videos[video_name]["recorded"] = True
                    else:
                        video_info["steps"] += 1
                else:
                    self.saved_videos[video_name]["size"] = video_size_kb
                    self.saved_videos[video_name]["steps"] = 0

    def add_video(self, video_path: str, log_name: str = "Video"):
        self.run.log({log_name: wandb.Video(video_path, format="mp4")})

    def callback(self, step):
        self.log_video_files()
