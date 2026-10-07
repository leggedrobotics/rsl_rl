# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO checkpoints must preserve the exploration schedule's elapsed steps."""

from __future__ import annotations

import copy
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


def _algorithm(schedule: dict | None = None, with_rnd: bool = True) -> tuple[PPO, TensorDict]:
    """Build actual PPO and RND components for checkpoint round trips."""
    observations = TensorDict({"policy": torch.zeros(3, 2)}, batch_size=[3])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        observations,
        groups,
        "actor",
        2,
        hidden_dims=[3],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(observations, groups, "critic", 1, hidden_dims=[3])
    storage = RolloutStorage("rl", 3, 4, observations, [2])
    cfg = {
        "num_states": 2,
        "obs_groups": {"rnd_state": ["policy"]},
        "num_outputs": 1,
        "predictor_hidden_dims": [3],
        "target_hidden_dims": [3],
        "weight": 2.0,
        "weight_schedule": schedule,
    }
    algorithm = PPO(actor, critic, storage, rnd_cfg=cfg if with_rnd else None, schedule="fixed")
    if with_rnd:
        with torch.no_grad():
            for parameter in algorithm.rnd.parameters():
                parameter.zero_()
            algorithm.rnd.target[-1].bias.fill_(2.0)
    return algorithm, observations


@pytest.mark.parametrize(
    "schedule,expected_weight",
    [
        ({"mode": "step", "final_step": 4, "final_value": 0.25}, 0.25),
        ({"mode": "linear", "initial_step": 1, "final_step": 6, "final_value": 0.25}, 0.95),
    ],
)
def test_ppo_rnd_resume_preserves_next_scheduled_reward(tmp_path: Path, schedule: dict, expected_weight: float) -> None:
    """Continue both schedules at the independently calculated next weight."""
    original, observations = _algorithm(schedule)
    for _ in range(3):
        original.rnd.get_intrinsic_reward(observations)
    path = tmp_path / "ppo.pt"
    torch.save(original.save(), path)
    restored, restored_observations = _algorithm(schedule)
    restored.load(torch.load(path, map_location="cpu", weights_only=True), load_cfg=None, strict=True)
    assert restored.rnd.update_counter == 3
    assert isinstance(restored.rnd.update_counter, int)
    expected = torch.full((3,), 2.0 * expected_weight)
    torch.testing.assert_close(original.rnd.get_intrinsic_reward(observations), expected)
    torch.testing.assert_close(restored.rnd.get_intrinsic_reward(restored_observations), expected)
    assert restored.rnd.update_counter == original.rnd.update_counter == 4


def test_legacy_ppo_checkpoint_without_rnd_counter_loads_strictly() -> None:
    """Load older checkpoints without adding required RND model-state keys."""
    original, _ = _algorithm()
    checkpoint = copy.deepcopy(original.save())
    checkpoint.pop("rnd_update_counter", None)
    restored, observations = _algorithm()
    restored.load(checkpoint, load_cfg=None, strict=True)
    assert restored.rnd.update_counter == 0
    torch.testing.assert_close(restored.rnd.get_intrinsic_reward(observations), torch.full((3,), 4.0))


def test_selective_load_preserves_counter_when_rnd_is_not_loaded() -> None:
    """Retain the existing counter when loading only the actor."""
    source, observations = _algorithm()
    for _ in range(3):
        source.rnd.get_intrinsic_reward(observations)
    destination, _ = _algorithm()
    destination.rnd.update_counter = 7
    destination.load(source.save(), load_cfg={"actor": True, "rnd": False}, strict=True)
    assert destination.rnd.update_counter == 7


def test_no_rnd_checkpoint_remains_free_of_rnd_state() -> None:
    """Keep checkpoints unchanged for algorithms without RND."""
    source, _ = _algorithm(with_rnd=False)
    checkpoint = source.save()
    assert not any(key.startswith("rnd_") for key in checkpoint)
    restored, _ = _algorithm(with_rnd=False)
    restored.load(checkpoint, load_cfg=None, strict=True)


def test_rnd_model_state_remains_tensors_for_existing_broadcast_path() -> None:
    """Retain tensor-only model state for the distributed broadcast path."""
    source, observations = _algorithm()
    source.rnd.get_intrinsic_reward(observations)
    assert all(isinstance(value, torch.Tensor) for value in source.rnd.state_dict().values())
    # The elapsed step count is plain checkpoint metadata, not a per-step device tensor.
    assert source.save()["rnd_update_counter"] == 1
