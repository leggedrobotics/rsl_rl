# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for logger utilities."""

from __future__ import annotations

from rsl_rl.utils.logger import _format_duration


def test_format_duration_under_one_day() -> None:
    """Durations shorter than one day should use HH:MM:SS."""
    assert _format_duration(7 * 3600 + 2 * 60 + 16) == "07:02:16"


def test_format_duration_over_one_day() -> None:
    """Durations longer than one day should include the day count."""
    assert _format_duration(2 * 24 * 3600 + 22 * 3600 + 27 * 60 + 26) == "2 days, 22:27:26"
