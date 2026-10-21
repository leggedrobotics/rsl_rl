# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Logger."""

from __future__ import annotations

import pathlib
import torch

from rsl_rl.utils import LogWriter
from rsl_rl.utils.logger import Logger


class RecordingWriter(LogWriter):
    """LogWriter that records scalar tags and uploaded videos."""

    def __init__(self) -> None:  # noqa: D107
        self.tags: list[str] = []
        self.videos: list[tuple[pathlib.Path, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: D102
        self.tags.append(tag)

    def save_video(self, video: pathlib.Path, it: int) -> None:  # noqa: D102
        self.videos.append((video, it))


def _make_logger(log_dir: pathlib.Path) -> Logger:
    """Construct a Logger that writes to a RecordingWriter."""
    logger = Logger(
        log_dir=str(log_dir),
        cfg={"num_steps_per_env": 1, "algorithm": {}},
        env_cfg={},
        num_envs=1,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.writer = RecordingWriter()
    return logger


def _log(logger: Logger, it: int) -> None:
    """Log one iteration with placeholder metrics."""
    logger.log(
        it=it,
        start_it=0,
        total_it=it + 1,
        collect_time=0.1,
        learn_time=0.1,
        loss_dict={},
        learning_rate=1e-3,
        action_std=torch.ones(1),
        rnd_weight=None,
    )


class TestVideoUpload:
    """Tests for uploading videos from the log directory."""

    def test_each_video_is_uploaded_once(self, tmp_path: pathlib.Path) -> None:
        """Each video should be uploaded once, at the first iteration after it appears."""
        first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
        first.touch()
        logger = _make_logger(tmp_path)

        _log(logger, 0)
        second.touch()
        _log(logger, 1)
        _log(logger, 2)

        assert logger.writer.videos == [(first, 0), (second, 1)]


class TestTimeSeries:
    """Tests for the series logged against wall-clock time."""

    def test_log_writer_receives_no_time_series(self, tmp_path: pathlib.Path) -> None:
        """A LogWriter should only receive series indexed by iteration."""
        logger = _make_logger(tmp_path)
        logger.rewbuffer.append(1.0)
        logger.lenbuffer.append(10.0)

        _log(logger, 0)

        assert "Train/mean_reward" in logger.writer.tags
        assert not any(tag.endswith("/time") for tag in logger.writer.tags)
