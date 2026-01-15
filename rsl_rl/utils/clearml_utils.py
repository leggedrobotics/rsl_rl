# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    from clearml import Task
except ModuleNotFoundError:
    raise ModuleNotFoundError("clearml package is required to log to ClearML.") from None


class ClearmlSummaryWriter(SummaryWriter):
    """Summary writer for ClearML."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get ClearML task
        try:
            project_name = cfg["clearml_project"]
        except KeyError:
            raise KeyError("Please specify clearml_project in the runner config, e.g. `legged_gym`.") from None

        # Initialize ClearML Task
        self.task = Task.init(
            project_name=project_name, task_name=run_name, auto_connect_frameworks={"tensorboard": False}
        )

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        runner_cfg = dict(train_cfg)
        runner_cfg.pop("policy", None)
        runner_cfg.pop("algorithm", None)

        if isinstance(env_cfg, dict):
            env_dict = env_cfg
        else:
            env_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)

        self.task.connect(runner_cfg, name="runner_cfg")
        self.task.connect(train_cfg.get("policy", {}), name="policy_cfg")
        self.task.connect(train_cfg.get("algorithm", {}), name="alg_cfg")
        self.task.connect(env_dict, name="env_cfg")

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
        self.task.get_logger().report_scalar(tag, "series", scalar_value, iteration=global_step)

    def stop(self) -> None:
        self.task.close()

    def save_model(self, model_path: str, it: int) -> None:
        self.task.upload_artifact(name=f"model_{it}", artifact_object=model_path)

    def save_file(self, path: str) -> None:
        self.task.upload_artifact(name=os.path.basename(path), artifact_object=path)
