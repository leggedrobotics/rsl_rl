# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SAC and the standard replay interaction contract."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import SAC
from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner
from tests.runners.test_on_policy_runner import DummyEnv


def _cfg() -> dict:
    return {
        "num_steps_per_env": 3,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "SquashedGaussianDistribution"},
        },
        "critic": {"class_name": "QMLPModel", "hidden_dims": [16, 16], "obs_normalization": True},
        "algorithm": {"class_name": "SAC", "batch_size": 8, "learning_starts": 0, "replay_buffer_size": 40},
    }


def _algorithm() -> SAC:
    cfg = _cfg()
    cfg["multi_gpu"] = None
    env = DummyEnv()
    return SAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for replay parity"),
        ),
    ],
)
@pytest.mark.parametrize("scenario", ["boundaries", "wrapped", "independent"])
def test_n_step_targets_at_episode_boundaries(device: str, scenario: str) -> None:
    """N-step returns stop at each episode boundary and bootstrap partial windows from their last successor."""
    capacity, n_steps, rewards, ends, terminals, expected = {
        "boundaries": (
            10,
            3,
            [1, 2, 3, 4, 5, 6],
            [2, 4],
            [4],
            {
                0: (5.23, 102, 0.729),
                1: (4.7, 102, 0.81),
                2: (3, 102, 0.9),
                3: (8.5, 104, 0),
                4: (5, 104, 0),
                5: (6, 105, 0.9),
            },
        ),
        "wrapped": (
            4,
            2,
            [1, 2, 3, 4, 5, 6],
            [],
            [],
            {
                2: (6.6, 103, 0.81),
                3: (8.5, 104, 0.81),
                4: (10.4, 105, 0.81),
                5: (6, 105, 0.9),
            },
        ),
        "independent": (
            8,
            2,
            [1, 1, 1],
            [0],
            [0],
            {
                0: (1, 100, 0),
                1: (1.9, 102, 0.81),
                2: (1, 102, 0.9),
                10: (1.9, 111, 0.81),
                11: (1.9, 112, 0.81),
                12: (1, 112, 0.9),
            },
        ),
    }[scenario]
    env = DummyEnv(device=device)
    env.num_envs = 2 if scenario == "independent" else 1
    cfg = _cfg()
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(replay_buffer_size=capacity, n_steps=n_steps, gamma=0.9, batch_size=256)
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device)
    for step, reward in enumerate(rewards):
        start = step + torch.arange(env.num_envs, device=device, dtype=torch.float) * 10
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        done[0], terminated[0] = step in ends, step in terminals
        alg.storage.add(
            TensorDict(
                {
                    "observations": {"policy": start[:, None].expand(-1, 8)},
                    "actions": torch.zeros(env.num_envs, 4, device=device),
                    "rewards": torch.full((env.num_envs,), float(reward), device=device),
                    "next_observations": {"policy": (start + 100)[:, None].expand(-1, 8)},
                    "dones": done,
                    "terminated": terminated,
                },
                batch_size=[env.num_envs],
                device=device,
            )
        )
    torch.manual_seed(0)
    batch = alg._sample_batch()
    rows = (
        torch
        .stack(
            (
                batch.observations["policy"][:, 0],
                batch.rewards,
                batch.next_observations["policy"][:, 0],
                batch.bootstrap_discounts,
            ),
            dim=1,
        )
        .cpu()
        .tolist()
    )
    assert {int(row[0]) for row in rows} == expected.keys()
    for start, *values in rows:
        assert values == pytest.approx(expected[int(start)], abs=1e-6)


def test_terminal_observations_and_missing_final() -> None:
    """All done successors use final observations; only exclusive timeouts bootstrap."""
    alg = _algorithm()
    obs = DummyEnv().get_observations()
    dones = torch.tensor([True, True, False, False])
    alg.act(obs)
    with pytest.raises(ValueError, match="final_obs"):
        alg.process_env_step(obs, torch.ones(4), dones, {})
    alg.process_env_step(
        obs,
        torch.ones(4),
        dones,
        {
            "time_outs": torch.tensor([False, True, False, False]),
            "final_obs": TensorDict({"policy": torch.full_like(obs["policy"], 9.0)}, batch_size=[4]),
        },
    )
    torch.testing.assert_close(alg.storage.data["next_observations", "policy"][0, :2], torch.full((2, 8), 9.0))
    assert alg.storage.data["terminated"][0].tolist() == [True, False, False, False]
    assert alg.actor.obs_normalizer.count == 0
    alg.update()
    assert alg.actor.obs_normalizer.count == 4
    cfg = _cfg()
    cfg["multi_gpu"] = {"global_rank": 0, "local_rank": 0, "world_size": 2}
    with pytest.raises(NotImplementedError, match="Multi-GPU"):
        SAC.construct_algorithm(obs, DummyEnv(), cfg, "cpu")
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(replay_buffer_size=4, learning_starts=8)
    limited = SAC.construct_algorithm(obs, DummyEnv(), cfg, "cpu")
    limited.act(obs)
    with pytest.raises(ValueError, match="capacity"):
        limited.process_env_step(obs, torch.ones(4), torch.zeros(4), {})


@pytest.mark.parametrize("device", ["cpu"])
def test_standard_runner_and_direct_drive(tmp_path: Path, device: str) -> None:
    """Logged uniform warm-up, fresh resume, updates and exports use the inherited loop or direct calls."""
    from onnx.reference import ReferenceEvaluator

    assert OffPolicyRunner.learn is OnPolicyRunner.learn
    cfg = _cfg()
    cfg["algorithm"].update(learning_starts=16, num_updates=4, actor_update_interval=1)
    log_dir = tmp_path / "warmup"
    runner = OffPolicyRunner(DummyEnv(device=device), cfg, log_dir=str(log_dir), device=device)
    before = [p.detach().clone() for p in runner.alg.actor.parameters()]
    runner.learn(1)
    assert runner.alg.storage.size == 12 < runner.alg.learning_starts
    assert runner.alg.num_critic_updates == 0
    assert all(torch.equal(a, b) for a, b in zip(before, runner.alg.actor.parameters()))
    assert runner.alg.get_policy().output_std is not None
    assert list(log_dir.glob("events*"))
    assert not runner.alg.storage.data["actions"].is_inference()
    path = str(tmp_path / "sac.pt")
    runner.save(path)
    restored = OffPolicyRunner(DummyEnv(device=device), cfg, log_dir=str(tmp_path / "resume"), device=device)
    restored.load(path, load_optimizer=False)
    assert not restored.alg.actor_optimizer.state
    assert restored.alg._actor_loaded
    models = [restored.alg.actor, *restored.alg.critics]
    before_updates = [[p.detach().clone() for p in model.parameters()] for model in models]
    restored.learn(2)
    assert restored.alg.num_critic_updates == 4
    for model, parameters in zip(models, before_updates):
        assert any(not torch.equal(old, current) for old, current in zip(parameters, model.parameters()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    sample = restored.env.get_observations()[:1]
    with torch.inference_mode():
        expected = restored.alg.get_policy()(sample).cpu()
    restored.export_policy_to_jit(str(tmp_path))
    restored.export_policy_to_onnx(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(sample["policy"].cpu()), expected)
    evaluator = ReferenceEvaluator(str(tmp_path / "policy.onnx"))
    actual = evaluator.run(None, {evaluator.input_names[0]: sample["policy"].cpu().numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected)
    restored.save(path)
    assert restored.alg.actor_optimizer.state
    fresh = OffPolicyRunner(DummyEnv(device=device), cfg, device=device)
    fresh.load(path, load_optimizer=False)
    assert not any(
        opt.state for opt in (fresh.alg.actor_optimizer, fresh.alg.critic_optimizer, fresh.alg.alpha_optimizer)
    )
    cfg["multi_gpu"] = None
    env = DummyEnv(device=device)
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device)
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(5):
            obs, rewards, dones, extras = env.step(alg.act(obs))
            alg.process_env_step(obs, rewards, dones, extras)
        alg.compute_returns(obs)
    losses = alg.update()
    assert losses and all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert alg.num_critic_updates == 4
