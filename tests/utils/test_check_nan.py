# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for environment output validation."""

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.utils import check_nan, find_nonfinite, raise_if_nonfinite


def test_check_nan_accepts_finite_outputs() -> None:
    """Finite environment outputs should pass validation."""
    obs = TensorDict({"policy": torch.tensor([[0.0, 1.0]])}, batch_size=[1])
    rewards = torch.tensor([1.0])
    dones = torch.tensor([False])

    check_nan(obs, rewards, dones)


@pytest.mark.parametrize(
    ("invalid_values", "expected_value_name"),
    [
        ([torch.nan, 0.0], "NaN values"),
        ([torch.inf, 0.0], "Inf values"),
        ([-torch.inf, 0.0], "Inf values"),
        ([torch.nan, torch.inf], "NaN and Inf values"),
    ],
)
@pytest.mark.parametrize(
    ("output", "expected_source"),
    [
        ("observations", "observation group 'policy'"),
        ("rewards", "rewards"),
        ("dones", "dones"),
    ],
)
def test_check_nan_rejects_non_finite_outputs(
    invalid_values: list[float], expected_value_name: str, output: str, expected_source: str
) -> None:
    """Non-finite outputs should raise with their source and value kind."""
    obs = TensorDict({"policy": torch.zeros(1, 2)}, batch_size=[1])
    rewards = torch.zeros(2)
    dones = torch.zeros(2)
    invalid = torch.tensor(invalid_values)

    if output == "observations":
        obs["policy"][0] = invalid
    elif output == "rewards":
        rewards[:] = invalid
    else:
        dones[:] = invalid

    with pytest.raises(ValueError) as exc_info:
        check_nan(obs, rewards, dones)

    message = str(exc_info.value)
    assert expected_source in message
    assert expected_value_name in message


def test_check_nan_rejects_non_finite_final_observations_of_done_environments() -> None:
    """Final observations are checked only for environments that are done."""
    obs = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])
    rewards = torch.zeros(2)
    final_obs = TensorDict({"policy": torch.tensor([[0.0, 0.0], [torch.nan, 0.0]])}, batch_size=[2])

    check_nan(obs, rewards, torch.tensor([True, False]), {"final_obs": final_obs})
    with pytest.raises(ValueError, match=r"final observation group 'policy' .* NaN values"):
        check_nan(obs, rewards, torch.tensor([False, True]), {"final_obs": final_obs})


def test_nonfinite_flags_name_their_sources() -> None:
    """Deferred flags ignore final observations of running environments and name every non-finite source."""
    obs = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])
    final_obs = TensorDict({"policy": torch.tensor([[0.0, 0.0], [torch.inf, 0.0]])}, batch_size=[2])

    raise_if_nonfinite(find_nonfinite(obs, torch.zeros(2), torch.tensor([True, False]), {"final_obs": final_obs}))
    flags = find_nonfinite(obs, torch.tensor([0.0, torch.nan]), torch.tensor([False, True]), {"final_obs": final_obs})
    with pytest.raises(ValueError, match="rewards, final observation group 'policy'"):
        raise_if_nonfinite(flags)


def test_nonfinite_flags_check_every_row_for_every_step_final_observations() -> None:
    """With final_obs_every_step, final observations of running environments are checked too."""
    obs = TensorDict({"policy": torch.zeros(2, 2)}, batch_size=[2])
    final_obs = TensorDict({"policy": torch.tensor([[0.0, 0.0], [torch.nan, 0.0]])}, batch_size=[2])
    extras = {"final_obs": final_obs, "final_obs_every_step": True}
    with pytest.raises(ValueError, match="final observation group 'policy'"):
        raise_if_nonfinite(find_nonfinite(obs, torch.zeros(2), torch.tensor([True, False]), extras))
