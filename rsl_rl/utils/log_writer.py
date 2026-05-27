# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod


class LogWriter(ABC):
    """Abstract base class for logging backends.

    Log writers are configured via ``cfg["logger"]``, a dict with a ``"class_name"`` key pointing to the subclass and
    any additional keys forwarded as constructor kwargs. The class is resolved via
    :func:`~rsl_rl.utils.resolve_callable`. Only :meth:`add_scalar` must be implemented; all other methods are no-ops.
    """

    @abstractmethod
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Log a scalar metric.

        Args:
            tag: Name of the metric.
            scalar_value: Value of the metric.
            global_step: Current training iteration.
        """

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration. Called once at training start."""

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint."""

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file."""

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video file."""

    def stop(self) -> None:
        """Finalize and close the logging run."""
