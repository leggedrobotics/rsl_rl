# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for environment output validation."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.utils import find_nonfinite, raise_if_nonfinite


def test_nonfinite_flags_name_their_sources() -> None:
    """Deferred flags ignore final observations of running environments and name every non-finite source."""
    obs = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])
    final_obs = TensorDict({"policy": torch.tensor([[0.0, 0.0], [torch.inf, 0.0]])}, batch_size=[2])

    raise_if_nonfinite(find_nonfinite(obs, torch.zeros(2), torch.tensor([True, False]), {"final_obs": final_obs}))
    flags = find_nonfinite(obs, torch.tensor([0.0, torch.nan]), torch.tensor([False, True]), {"final_obs": final_obs})
    with pytest.raises(ValueError, match="rewards, final observation group 'policy'"):
        raise_if_nonfinite(flags)
