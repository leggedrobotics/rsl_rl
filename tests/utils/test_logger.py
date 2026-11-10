# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Logger."""

from __future__ import annotations

import pathlib
import torch

import pytest

from rsl_rl.utils import LogWriter
from rsl_rl.utils.logger import Logger


class RecordingWriter(LogWriter):
    """LogWriter that records scalar tags and uploaded videos."""

    def __init__(self) -> None:  # noqa: D107
        self.tags: list[str] = []
        self.scalars: dict[str, float] = {}
        self.videos: list[tuple[pathlib.Path, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:  # noqa: D102
        self.tags.append(tag)
        self.scalars[tag] = scalar_value

    def save_video(self, video: pathlib.Path, it: int) -> None:  # noqa: D102
        self.videos.append((video, it))


def _make_logger(log_dir: pathlib.Path, num_envs: int = 1) -> Logger:
    """Construct a Logger that writes to a RecordingWriter."""
    logger = Logger(
        log_dir=str(log_dir),
        cfg={"num_steps_per_env": 1, "algorithm": {}},
        env_cfg={},
        num_envs=num_envs,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    logger.writer = RecordingWriter()
    return logger


def _log(
    logger: Logger,
    it: int,
    loss_dict: dict[str, float] | None = None,
    stats_dict: dict[str, float] | None = None,
) -> None:
    """Log one iteration with placeholder metrics."""
    logger.log(
        it=it,
        start_it=0,
        total_it=it + 1,
        collect_time=0.1,
        learn_time=0.1,
        loss_dict=loss_dict or {},
        stats_dict=stats_dict or {},
    )


def test_statistics_are_separate_from_losses(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Statistics retain their tags and console labels without being treated as optimization losses."""
    logger = _make_logger(tmp_path)
    stats_dict = {
        "Model/mean_policy_entropy": 0.2,
        "Model/mean_critic_q_value": 3.0,
        "Optimizer/learning_rate": 1e-3,
        "Model/mean_policy_std": 1.0,
        "Rnd/weight": 0.5,
        "Custom/statistic": 4.0,
        "statistic": 5.0,
        "Custom/nested/statistic": 6.0,
    }
    _log(logger, 0, {"actor": 1.0}, stats_dict)

    assert {
        "Loss/actor",
        "Performance/total_fps",
        "Performance/collection_time",
        "Performance/learning_time",
        *stats_dict,
    } <= set(logger.writer.tags)
    assert not any(tag.startswith("Stats/") for tag in logger.writer.tags)
    assert not {"Loss/entropy", "Loss/q_value"} & set(logger.writer.tags)
    output = capsys.readouterr().out
    assert "Mean actor loss:" in output
    for name in (
        "Mean policy entropy",
        "Mean critic q value",
        "Learning rate",
        "Mean policy std",
        "Weight",
        "Statistic",
        "Nested/statistic",
    ):
        assert f"{name}:" in output
    for category in ("Model/", "Loss/", "Optimizer/", "Rnd/", "Custom/"):
        assert category not in output
    assert "entropy loss:" not in output
    assert "q_value loss:" not in output


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


def test_step_metrics_accumulate_over_completed_episodes(tmp_path: pathlib.Path) -> None:
    """Custom metrics preserve partial episodes across logging calls and reset environments independently."""
    logger = _make_logger(tmp_path, num_envs=2)
    with torch.inference_mode():
        logger.process_env_step(
            torch.tensor([False, True]),
            {},
            {"Training/mean_reward": torch.tensor([1.0, 10.0]), "Custom/total": torch.tensor([2.0, 20.0])},
        )
    _log(logger, 0)
    assert logger.writer.scalars["Training/mean_reward"] == 10.0
    assert logger.writer.scalars["Custom/total"] == 20.0

    logger.process_env_step(
        torch.tensor([False, False]),
        {},
        {"Training/mean_reward": torch.tensor([3.0, 30.0]), "Custom/total": torch.tensor([4.0, 40.0])},
    )
    _log(logger, 1)
    assert logger.writer.scalars["Custom/total"] == 20.0

    logger.process_env_step(
        torch.tensor([True, True]),
        {},
        {"Training/mean_reward": torch.tensor([5.0, 50.0]), "Custom/total": torch.tensor([6.0, 60.0])},
    )
    _log(logger, 2)
    assert logger.writer.scalars["Training/mean_reward"] == 33.0
    assert logger.writer.scalars["Custom/total"] == 44.0
    assert logger.writer.scalars["Training/mean_episode_length"] == 2.0
    torch.testing.assert_close(logger.cur_step_sums["Custom/total"], torch.zeros(2))


class TestTimeSeries:
    """Tests for the series logged against wall-clock time."""

    def test_log_writer_receives_no_time_series(self, tmp_path: pathlib.Path) -> None:
        """A LogWriter should only receive series indexed by iteration."""
        logger = _make_logger(tmp_path)
        logger.process_env_step(torch.ones(1), {}, {"Training/mean_reward": torch.ones(1)})

        _log(logger, 0)

        assert "Training/mean_reward" in logger.writer.tags
        assert not any(tag.endswith("/time") for tag in logger.writer.tags)
