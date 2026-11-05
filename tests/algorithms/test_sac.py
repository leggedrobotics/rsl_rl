# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for SAC and the standard replay interaction contract."""

from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict
from types import SimpleNamespace

import pytest

from rsl_rl.algorithms import SAC
from rsl_rl.algorithms.sac import _Batch
from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner
from tests.runners.test_on_policy_runner import DummyEnv


def _make_train_cfg() -> dict:
    """Create a SAC configuration with small networks for testing."""
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
        "algorithm": {"class_name": "SAC", "batch_size": 8, "num_warmup_transitions": 0, "replay_buffer_capacity": 40},
    }


def _build_sac() -> SAC:
    """Build a SAC instance with small networks for testing."""
    cfg = _make_train_cfg()
    cfg["multi_gpu"] = None
    env = DummyEnv()
    return SAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")


@pytest.mark.parametrize(
    ("actor", "critic", "image_only", "categorical", "device"),
    [
        ("CNNModel", "QMLPModel", False, False, "cpu"),
        ("MLPModel", "QCNNModel", False, False, "cpu"),
        ("CNNModel", "QCNNModel", True, True, "cpu"),
        pytest.param(
            "CNNModel",
            "QCNNModel",
            False,
            True,
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CNN graph capture"),
        ),
    ],
)
def test_cnn_sac_training_and_export(
    actor: str, critic: str, image_only: bool, categorical: bool, device: str, tmp_path: Path
) -> None:
    """Preserve image replay, learn both encoders and Q action gradients, and round-trip policy exports."""
    from onnx.reference import ReferenceEvaluator

    cfg = _make_train_cfg()
    cfg["algorithm"].update(
        critic_loss_type="categorical" if categorical else "mse",
        num_bins=11,
        num_updates=6,
        num_return_steps=3,
        use_cuda_graph=device != "cpu",
    )
    for role, model in (("actor", actor), ("critic", critic)):
        cfg[role]["class_name"] = model
        if "CNN" in model:
            cfg["obs_groups"][role] = ["image"] if image_only else ["policy", "image"]
            cfg[role]["cnn_cfg"] = {"output_channels": [4], "kernel_size": 3, "stride": 2, "flatten": True}
        if categorical:
            cfg[role]["network_cfg"] = {"class_name": "ResidualMLP", "hidden_dim": 16, "num_blocks": 1, "expansion": 2}
    env = DummyEnv(device=device, include_image=True)
    runner = OffPolicyRunner(env, cfg, device=device)
    alg = runner.alg
    obs = env.get_observations()
    final = obs.clone()
    final["image"].fill_(7.0)
    dones = torch.tensor([True, True, False, False], device=device)
    alg.act(obs)
    alg.process_env_step(obs, torch.ones(4, device=device), dones, {"final_obs": final, "time_outs": dones})
    torch.testing.assert_close(alg.storage.next_observations["image"][0, :2], final["image"][:2])
    encoders = [model.cnns for model in [alg.actor, *alg.critics] if hasattr(model, "cnns")]
    before = [[parameter.detach().clone() for parameter in encoder.parameters()] for encoder in encoders]
    runner.learn(1)
    for encoder, old in zip(encoders, before):
        assert any(not torch.equal(value, parameter) for value, parameter in zip(old, encoder.parameters()))
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in encoder.parameters()
        )
    batch = alg._sample_batch()
    assert batch.observations["image"].shape == (8, 1, 16, 16)
    assert batch.next_observations["image"].shape == (8, 1, 16, 16)
    if device != "cpu":
        assert alg._graph_updates
    assert all(torch.isfinite(torch.tensor(value)) for value in alg.update().values())
    alg.eval_mode()
    actions = alg.actor(obs)
    q = alg._expected_value(alg.critics[0](obs, alg._normalize_actions(actions)))
    parameters = list(alg.actor.cnns.parameters()) if hasattr(alg.actor, "cnns") else list(alg.actor.parameters())
    gradients = torch.autograd.grad(q.sum(), [actions, *parameters])
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0].abs().sum() > 0
    assert any(gradient.abs().sum() > 0 for gradient in gradients[1:])
    expected = alg.get_policy()(obs).detach().cpu()
    runner.save(str(tmp_path / "cnn.pt"))
    restored = OffPolicyRunner(env, cfg, device=device)
    restored.load(str(tmp_path / "cnn.pt"), map_location=device)
    restored.alg.eval_mode()
    torch.testing.assert_close(restored.alg.actor(obs).cpu(), expected)
    for source, target in zip(alg.critics, restored.alg.critics):
        torch.testing.assert_close(source(obs, actions), target(obs, actions))
    runner.export_policy_to_jit(str(tmp_path))
    runner.export_policy_to_onnx(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    vector = torch.empty(4, 0) if image_only else obs["policy"].cpu()
    actual = scripted(vector, [obs["image"].cpu()]) if actor == "CNNModel" else scripted(vector)
    torch.testing.assert_close(actual, expected)
    evaluator = ReferenceEvaluator(str(tmp_path / "policy.onnx"))
    inputs = {"obs": vector[:1].numpy(), "image": obs["image"][:1].cpu().numpy()}
    actual = evaluator.run(None, {name: inputs[name] for name in evaluator.input_names})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected[:1], atol=1e-5, rtol=1e-5)

    final["image"][0] = float("nan")
    restored.alg.act(obs)
    restored.alg.process_env_step(obs, torch.ones(4, device=device), dones, {"final_obs": final})
    with pytest.raises(ValueError, match="nonfinite final_obs"):
        restored.alg.update()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture"),
        ),
    ],
)
def test_standard_runner_and_direct_drive(tmp_path: Path, device: str) -> None:
    """Logged uniform warm-up, fresh resume, updates and exports use the inherited loop or direct calls."""
    from onnx.reference import ReferenceEvaluator

    assert OffPolicyRunner.learn is OnPolicyRunner.learn
    cfg = _make_train_cfg()
    cfg["algorithm"].update(
        num_warmup_transitions=16, num_updates=4, actor_update_interval=1, use_cuda_graph=device != "cpu"
    )
    log_dir = tmp_path / "warmup"
    runner = OffPolicyRunner(DummyEnv(device=device), cfg, log_dir=str(log_dir), device=device)
    before = [p.detach().clone() for p in runner.alg.actor.parameters()]
    runner.learn(1)
    assert runner.alg.storage.size == 12 < runner.alg.num_warmup_transitions
    assert runner.alg.num_critic_updates == 0
    assert all(torch.equal(a, b) for a, b in zip(before, runner.alg.actor.parameters()))
    assert runner.alg.get_policy().output_std is not None
    assert list(log_dir.glob("events*"))
    assert not runner.alg.storage.actions.is_inference()
    path = str(tmp_path / "sac.pt")
    runner.save(path)
    restored = OffPolicyRunner(DummyEnv(device=device), cfg, log_dir=str(tmp_path / "resume"), device=device)
    load_cfg = restored.alg.get_default_load_cfg()
    load_cfg["optimizer"] = False
    restored.load(path, load_cfg=load_cfg)
    assert not restored.alg.actor_optimizer.state
    assert restored.alg._actor_loaded
    models = [restored.alg.actor, *restored.alg.critics]
    before_updates = [[p.detach().clone() for p in model.parameters()] for model in models]
    restored.learn(2)
    assert restored.alg.num_critic_updates == 4
    for model, parameters in zip(models, before_updates):
        assert any(not torch.equal(old, current) for old, current in zip(parameters, model.parameters()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    if device != "cpu":
        assert 0 in restored.alg._graph_updates
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
    fresh.load(path, load_cfg=load_cfg)
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
    capacity, num_return_steps, rewards, ends, terminals, expected = {
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
    cfg = _make_train_cfg()
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(
        replay_buffer_capacity=capacity, num_return_steps=num_return_steps, gamma=0.9, batch_size=256
    )
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, device)
    for step, reward in enumerate(rewards):
        start = step + torch.arange(env.num_envs, device=device, dtype=torch.float) * 10
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        terminated = torch.zeros_like(done)
        done[0] = step in ends
        terminated[0] = step in terminals
        transition = alg.storage.Transition()
        transition.observations = TensorDict(
            {"policy": start[:, None].expand(-1, 8)}, batch_size=[env.num_envs], device=device
        )
        transition.actions = torch.zeros(env.num_envs, 4, device=device)
        transition.rewards = torch.full((env.num_envs,), float(reward), device=device)
        transition.next_observations = TensorDict(
            {"policy": (start + 100)[:, None].expand(-1, 8)}, batch_size=[env.num_envs], device=device
        )
        transition.dones = done
        transition.terminated = terminated
        alg.storage.add_transition(transition)
    torch.manual_seed(0)
    batch = alg._sample_batch()
    rows = (
        torch.stack(
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
    alg = _build_sac()
    obs = DummyEnv().get_observations()
    dones = torch.tensor([True, True, False, False])
    alg.act(obs)
    alg.process_env_step(obs, torch.ones(4), dones, {})
    with pytest.raises(ValueError, match="final_obs"):
        alg.update()
    alg = _build_sac()
    alg.act(obs)
    alg.process_env_step(
        obs,
        torch.ones(4),
        dones,
        {
            "time_outs": torch.tensor([False, True, False, False]),
            "final_obs": {"policy": torch.full_like(obs["policy"], 9.0)},
        },
    )
    torch.testing.assert_close(alg.storage.next_observations["policy"][0, :2], torch.full((2, 8), 9.0))
    assert alg.storage.terminated[0].tolist() == [True, False, False, False]
    assert alg.actor.obs_normalizer.count == 0
    alg.update()
    assert alg.actor.obs_normalizer.count == 4
    cfg = _make_train_cfg()
    cfg["multi_gpu"] = {"global_rank": 0, "local_rank": 0, "world_size": 2}
    with pytest.raises(ValueError, match="initialized process group"):
        SAC.construct_algorithm(obs, DummyEnv(), cfg, "cpu")
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(replay_buffer_capacity=4, num_warmup_transitions=8)
    limited = SAC.construct_algorithm(obs, DummyEnv(), cfg, "cpu")
    limited.act(obs)
    with pytest.raises(ValueError, match="capacity"):
        limited.process_env_step(obs, torch.ones(4), torch.zeros(4), {})


@pytest.mark.parametrize("critic_loss_type", ["mse", "categorical"])
def test_terminal_targets_ignore_non_finite_successor_values(critic_loss_type: str) -> None:
    """Rows without bootstrapping produce finite targets even if the successor values are not finite."""
    cfg = _make_train_cfg()
    cfg["multi_gpu"] = None
    cfg["algorithm"]["critic_loss_type"] = critic_loss_type
    env = DummyEnv()
    alg = SAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    output_dim = 1 if critic_loss_type == "mse" else alg.num_bins
    next_outputs = [torch.full((2, output_dim), torch.nan), torch.full((2, output_dim), torch.nan)]
    batch = SimpleNamespace(rewards=torch.ones(2), bootstrap_discounts=torch.zeros(2))
    target = alg._compute_target(next_outputs, batch, torch.full((2,), torch.nan))  # type: ignore[arg-type]
    assert torch.isfinite(target).all()
    values = target if critic_loss_type == "mse" else (target * alg._bin_values).sum(-1)
    torch.testing.assert_close(values, torch.ones(2))


@pytest.mark.parametrize(("network", "mixed_precision"), [(None, False), ("ResidualMLP", False), (None, True)])
def test_opt_in_flash_components_and_export(network: str | None, mixed_precision: bool, tmp_path: Path) -> None:
    """Categorical updates, return scaling, held noise and residual export work through the standard runner."""
    from onnx.reference import ReferenceEvaluator

    cfg = _make_train_cfg()
    cfg["algorithm"].update(
        critic_loss_type="categorical",
        use_mixed_precision=mixed_precision,
        num_bins=11,
        target_entropy_std=0.15,
        reward_normalization=True,
        exploration_type="held",
        learning_rate_schedule="cosine",
        num_schedule_updates=10,
        num_updates=2,
    )
    if network:
        for role in ("actor", "critic"):
            cfg[role]["network_cfg"] = {"class_name": network, "hidden_dim": 16, "num_blocks": 1, "expansion": 2}
    runner = OffPolicyRunner(DummyEnv(), cfg)
    runner.learn(3)
    assert runner.alg.num_critic_updates == 6
    assert runner.alg._return_normalizer.count == 36
    sample = runner.env.get_observations()
    runner.alg.eval_mode()
    expected = runner.alg.get_policy()(sample)
    runner.export_policy_to_jit(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(sample["policy"]), expected)
    if network == "ResidualMLP":
        runner.export_policy_to_onnx(str(tmp_path))
        evaluator = ReferenceEvaluator(str(tmp_path / "policy.onnx"))
        actual = evaluator.run(None, {evaluator.input_names[0]: sample["policy"][:1].numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), expected[:1])
    checkpoint = runner.alg.save()
    restored = OffPolicyRunner(DummyEnv(), cfg)
    restored.alg.load(checkpoint, None, True)
    assert restored.alg.learning_rate == runner.alg.learning_rate < cfg["algorithm"].get("critic_learning_rate", 1e-4)


def _mirror(
    env: object, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Mirror all inputs around zero while retaining the original batch first."""
    return (
        torch.cat([obs, -obs]) if obs is not None else None,
        torch.cat([actions, -actions]) if actions is not None else None,
    )


@pytest.mark.parametrize("augmentation", [False, True])
def test_symmetry_replay_endpoints_and_actor_loss(augmentation: bool) -> None:
    """Symmetry handles replay endpoint pairs and the existing auxiliary actor loss."""
    cfg = _make_train_cfg()
    cfg["algorithm"]["symmetry_cfg"] = {
        "data_augmentation_func": _mirror,
        "use_data_augmentation": augmentation,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
    }
    runner = OffPolicyRunner(DummyEnv(), cfg)
    runner.learn(2)
    batch = runner.alg._sample_batch()
    if augmentation:
        batch_size = runner.alg.batch_size
        torch.testing.assert_close(
            batch.observations["policy"][batch_size:], -batch.observations["policy"][:batch_size]
        )
        torch.testing.assert_close(
            batch.next_observations["policy"][batch_size:], -batch.next_observations["policy"][:batch_size]
        )
        torch.testing.assert_close(batch.actions[batch_size:], -batch.actions[:batch_size])
    assert torch.isfinite(torch.tensor(runner.alg.update()["symmetry"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_cuda_graph_delayed_actor_and_resume() -> None:
    """The captured update graph can train, reload eager state and resume under the inherited runner."""
    cfg = _make_train_cfg()
    cfg["algorithm"].update(
        use_cuda_graph=True,
        num_updates=12,
        actor_update_interval=2,
        learning_rate_schedule="cosine",
        num_schedule_updates=100,
        critic_loss_type="categorical",
        num_bins=11,
    )
    for role in ("actor", "critic"):
        cfg[role]["network_cfg"] = {"class_name": "ResidualMLP", "hidden_dim": 16, "num_blocks": 1, "expansion": 2}
    runner = OffPolicyRunner(DummyEnv(device="cuda:0"), cfg, device="cuda:0")
    runner.learn(4)
    assert all(graph.graph is not None for graph in runner.alg._graph_updates.values())
    checkpoint = runner.alg.save()
    runner.alg.load(checkpoint, None, True)
    runner.learn(1)
    assert runner.alg.num_critic_updates == 60
    assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())


@pytest.mark.parametrize("timeout", [False, True])
def test_nonfinite_final_observations_rejected_before_training(timeout: bool) -> None:
    """Validate only consumed final rows, including both hard terminals and timeouts, before any update."""
    alg = _build_sac()
    obs = DummyEnv().get_observations()
    final = obs.clone()
    final["policy"][0] = float("nan")
    dones = torch.tensor([True, False, False, False])
    alg.act(obs)
    alg.process_env_step(obs, torch.ones(4), dones, {"final_obs": final, "time_outs": dones & timeout})
    before = [p.detach().clone() for p in alg.actor.parameters()]
    with pytest.raises(ValueError, match=r"nonfinite final_obs.*policy"):
        alg.update()
    assert all(torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))
    alg = _build_sac()
    final["policy"][0] = 7.0
    final["policy"][1:] = float("nan")
    alg.act(obs)
    alg.process_env_step(obs, torch.ones(4), dones, {"final_obs": final, "time_outs": dones & timeout})
    torch.testing.assert_close(alg.storage.next_observations["policy"][0, 0], torch.full((8,), 7.0))
    assert alg.storage.terminated[0, 0].item() == (not timeout)
    assert torch.isfinite(alg.storage.next_observations["policy"][0]).all()


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("batch_norm", [False, True])
@pytest.mark.parametrize("all_terminal", [False, True])
def test_terminal_successors_never_poison_model_forwards(
    categorical: bool, batch_norm: bool, all_terminal: bool
) -> None:
    """Terminal NaNs must not reach actor, critic, or running batch statistics."""
    cfg = _make_train_cfg()
    cfg["multi_gpu"] = None
    cfg["algorithm"].update(critic_loss_type="categorical" if categorical else "mse", num_bins=11)
    if batch_norm:
        for role in ("actor", "critic"):
            cfg[role]["network_cfg"] = {"class_name": "ResidualMLP", "hidden_dim": 16, "num_blocks": 1, "expansion": 2}
    env = DummyEnv()
    obs = env.get_observations()
    alg = SAC.construct_algorithm(obs, env, cfg, "cpu")
    successors = obs.clone()
    terminals = torch.ones(4, dtype=torch.bool) if all_terminal else torch.tensor([True, False, True, False])
    successors["policy"][terminals] = float("nan")
    successors["policy"][~terminals] = 9.0
    if not all_terminal:
        successors["policy"][2] = 7.0
    seen = []

    def check_inputs(module: torch.nn.Module, inputs: tuple) -> None:
        assert torch.isfinite(inputs[0]["policy"]).all()
        seen.append(inputs[0]["policy"].detach().clone())

    handles = [
        model.register_forward_pre_hook(check_inputs) for model in [alg.actor, *alg.critics, *alg.target_critics]
    ]
    batch = _Batch(
        observations=obs,
        actions=torch.zeros(4, 4),
        rewards=torch.ones(4),
        next_observations=successors,
        bootstrap_discounts=(~terminals).float() * alg.gamma,
    )
    losses = alg._gradient_step(batch, True)
    for handle in handles:
        handle.remove()
    assert seen
    if not all_terminal:
        assert any((value == 9.0).all(dim=-1).any() for value in seen)
        assert any((value == 7.0).all(dim=-1).any() for value in seen)
    assert all(torch.isfinite(value).all() for value in losses.values())
    for model in [alg.actor, *alg.critics, *alg.target_critics]:
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_sac_rejects_shared_model_parameters() -> None:
    """Separate optimizers and critic targets require independently owned encoders."""
    cfg = _make_train_cfg()
    for role, name in (("actor", "CNNModel"), ("critic", "QCNNModel")):
        cfg[role].update(class_name=name, cnn_cfg={"output_channels": [4], "kernel_size": 3})
        cfg["obs_groups"][role] = ["policy", "image"]
    alg = OffPolicyRunner(DummyEnv(include_image=True), cfg).alg
    alg.critics[0].cnns = alg.actor.cnns
    with pytest.raises(ValueError, match="separate parameters"):
        SAC(alg.actor, list(alg.critics), alg.storage)


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_recurrent_models_are_rejected_before_construction(role: str) -> None:
    """Reject recurrent configs explicitly instead of silently treating random replay as sequences."""
    cfg = _make_train_cfg()
    cfg[role]["class_name"] = "RNNModel"
    cfg["multi_gpu"] = None
    env = DummyEnv()
    with pytest.raises(ValueError, match="does not yet support recurrent models"):
        SAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")
