# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name)
        wandb.config.update({"log_dir": log_dir})

        # Initialize video logging
        self.video_files: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        wandb.config.update({"runner_cfg": train_cfg})
        wandb.config.update({"policy_cfg": train_cfg["policy"]})
        wandb.config.update({"alg_cfg": train_cfg["algorithm"]})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

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
        wandb.log({tag: scalar_value}, step=global_step)

    def add_video_files(self, log_dir: str | os.PathLike, step: int, fps: int = 30) -> None:
        log_path = Path(log_dir)

        # Prevent logging if dir not initialized
        if not log_path.exists():
            return

        for video_path in log_path.rglob("*.mp4"):
            video_name = video_path.name
            if video_name in self.video_files:
                continue

            self.video_files.add(video_name)

            try:
                video_artifact = wandb.Video(str(video_path), format="mp4")  # type: ignore[arg-type]
                wandb.log({"Video": video_artifact}, step=step)
            except Exception as e:
                print(f"\033[33mWarning: Failed to log video {video_name} to wandb: {e}\033[0m")

    def stop(self) -> None:
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))
