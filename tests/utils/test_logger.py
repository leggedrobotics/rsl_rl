# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the training logger."""

from __future__ import annotations

import copy
import pathlib

import pytest

import rsl_rl.utils.logger as logger_module
from rsl_rl.utils.log_writer import LogWriter
from rsl_rl.utils.logger import Logger


class RecordingLogWriter(LogWriter):
    """Minimal custom writer that records its constructor and config inputs."""

    def __init__(self, log_dir: str, project_name: str) -> None:
        """Initialize the writer with the forwarded custom logger arguments."""
        self.log_dir = log_dir
        self.project_name = project_name
        self.train_cfg: dict | None = None

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Implement the required scalar logging interface."""

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Record an independent snapshot of the uploaded training config."""
        self.train_cfg = copy.deepcopy(train_cfg)


def test_custom_log_writer_does_not_mutate_training_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Custom writer construction should preserve the caller-owned config."""
    train_cfg = {
        "algorithm": {"rnd_cfg": None},
        "logger": {"class_name": "RecordingLogWriter", "project_name": "config-regression"},
    }
    expected_cfg = copy.deepcopy(train_cfg)
    monkeypatch.setattr(logger_module, "resolve_callable", lambda _: RecordingLogWriter)
    monkeypatch.setattr(Logger, "_store_code_state", lambda _: [])

    logger = Logger(
        log_dir=str(tmp_path),
        cfg=train_cfg,
        env_cfg={},
        num_envs=1,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.init_logging_writer()

    assert train_cfg == expected_cfg
    assert isinstance(logger.writer, RecordingLogWriter)
    assert logger.writer.log_dir == str(tmp_path)
    assert logger.writer.project_name == "config-regression"
    assert logger.writer.train_cfg == expected_cfg
