# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod


class LogWriter(ABC):
    """Abstract base class for rsl_rl logging backends.

    Subclass this to implement a custom logging backend. Only :meth:`add_scalar`
    is required; all other methods are no-ops by default.

    Example::

        from rsl_rl.utils import LogWriter

        class MyWriter(LogWriter):
            def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
                db.insert(tag=tag, value=scalar_value, step=global_step)
    """

    @abstractmethod
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Log a scalar metric.

        Args:
            tag: Metric name using ``"Group/name"`` convention (e.g. ``"Train/mean_reward"``).
            scalar_value: The scalar value to record.
            global_step: Training iteration used as the x-axis.
        """

    @abstractmethod
    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration. Called once at training start."""

    @abstractmethod
    def save_model(self, model_path: str, it: int) -> None:
        """Upload or archive a model checkpoint."""

    @abstractmethod
    def save_file(self, path: str) -> None:
        """Upload or archive an arbitrary file (e.g., git diff)."""

    @abstractmethod
    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video file."""

    @abstractmethod
    def stop(self) -> None:
        """Finalize and close the logging run."""
