# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for resolve_mixed_precision."""

from __future__ import annotations

import pytest

from rsl_rl.utils import resolve_mixed_precision


def test_explicit_true_is_true() -> None:
    assert resolve_mixed_precision(True, "cuda") is True
    assert resolve_mixed_precision(True, "cpu") is True


def test_explicit_false_is_false() -> None:
    assert resolve_mixed_precision(False, "cuda") is False
    assert resolve_mixed_precision(False, "cpu") is False


def test_auto_on_cpu_is_false() -> None:
    assert resolve_mixed_precision("auto", "cpu") is False


def test_auto_on_cuda_follows_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_mixed_precision("auto", "cuda") is True
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_mixed_precision("auto", "cuda") is False


def test_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        resolve_mixed_precision("yes", "cuda")
