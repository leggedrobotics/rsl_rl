# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import swanlab
except ModuleNotFoundError:
    raise ModuleNotFoundError("swanlab package is required to log to Swanlab.") from None


class SwanlabSummaryWriter(SummaryWriter):
    """Summary writer for Swanlab."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get swanlab project and entity
        try:
            project = cfg["swanlab_project"]
        except KeyError:
            raise KeyError("Please specify swanlab_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["SWANLAB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize swanlab
        swanlab.init(project=project, entity=entity, name=run_name)
        swanlab.config.update({"log_dir": log_dir})

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        swanlab.config.update({"runner_cfg": train_cfg})
        swanlab.config.update({"policy_cfg": train_cfg["policy"]})
        swanlab.config.update({"alg_cfg": train_cfg["algorithm"]})
        try:
            swanlab.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            swanlab.config.update({"env_cfg": asdict(env_cfg)})

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
        swanlab.log({tag: scalar_value}, step=global_step)

    def stop(self) -> None:
        swanlab.finish()

    def save_model(self, model_path: str, it: int) -> None:
        raise NotImplementedError("Swanlab does not support saving files currently.")

    def save_file(self, path: str) -> None:
        raise NotImplementedError("Swanlab does not support saving files currently.")
