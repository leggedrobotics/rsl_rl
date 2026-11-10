# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic learning, replay boundaries and deployment on a synthetic VecEnv."""

from __future__ import annotations

import copy
import math
import torch
from pathlib import Path
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.algorithms import SuperMimic
from rsl_rl.algorithms.supermimic import lambda_returns
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, EuclideanStateRepresentation, State
from rsl_rl.env import VecEnv
from rsl_rl.modules import WorldModel
from rsl_rl.runners import OffPolicyRunner


class ScalarTask(DifferentiableTask):
    """A Euclidean system with privileged context and one successor event."""

    num_labels = 1

    def __init__(self, env: VecEnv, device: str) -> None:
        """Bind a one-dimensional Euclidean state."""
        self.representation = EuclideanStateRepresentation(1, error_floor=0.01)

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Expose position and the most recent action."""
        return torch.cat((state[0], prev_actions[..., -1, :]), -1)

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Use the explicit deployable goal unchanged."""
        return goal

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Expose privileged context and both action predecessors to dynamics."""
        return torch.cat((context, prev_actions.flatten(-2)), -1)

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Build privileged scalar-value inputs."""
        return torch.cat((state[0], goal, context, prev_actions.flatten(-2)), -1)

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold exogenous goals and context fixed."""
        return goal, context

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reward goal tracking and penalize action magnitude."""
        target = goal[..., :1] if goal.shape[-1] else torch.zeros_like(next_state[0])
        reward = 1 - (next_state[0] - target).square().squeeze(-1) - 0.01 * actions.square().sum(-1)
        return reward if labels is None else reward + 0.01 * labels[..., 0]

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Keep the synthetic physical system alive in imagination."""
        return torch.zeros_like(next_state[0][..., 0])


class ScalarEnv(VecEnv):
    """Owned, pre-reset observations with optional interval pushes and nonzero reset actions."""

    def __init__(
        self,
        device: str = "cpu",
        goal_dim: int = 1,
        context_dim: int = 1,
        reset_action: float = 0.0,
        pushes: bool = False,
    ) -> None:
        """Allocate a small vectorized system with explicit physical observation groups."""
        self.num_envs, self.num_actions = 4, 1
        self.device, self.cfg = device, {}
        self.max_episode_length = 5
        self.episode_length_buf = torch.zeros(4, dtype=torch.long, device=device)
        self.position = torch.zeros(4, 1, device=device)
        self.goal = torch.ones(4, goal_dim, device=device)
        self.context = torch.zeros(4, context_dim, device=device)
        self.previous = torch.full((4, 1), reset_action, device=device)
        self.previous_previous = self.previous.clone()
        self.reset_action, self.pushes = reset_action, pushes

    def get_observations(self) -> TensorDict:
        """Return owned physical observations and reported action predecessors."""
        return TensorDict(
            {
                "state": self.position.clone(),
                "goal": self.goal.clone(),
                "context": self.context.clone(),
                "labels": (self.position > 0).float(),
                "prev_action": self.previous.clone(),
                "prev_prev_action": self.previous_previous.clone(),
            },
            batch_size=[4],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance, capture physical successors, then apply pushes and resets."""
        self.position.add_(0.2 * actions)
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(actions)
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        final = self.get_observations()
        rewards = 1 - (self.position[:, 0] - 1).square()
        if self.pushes:
            self.position[~dones] += 10
        self.position[dones] = 0
        self.episode_length_buf[dones] = 0
        self.previous[dones] = self.reset_action
        self.previous_previous[dones] = self.reset_action
        return (
            self.get_observations(),
            rewards,
            dones,
            {"final_obs": final, "final_obs_every_step": True, "time_outs": dones.clone()},
        )


def _cfg(**options: object) -> dict:
    algorithm = {
        "class_name": "SuperMimic",
        "task_cfg": {"class_name": ScalarTask},
        "updates_per_step": 1.0,
        "replay_buffer_capacity": 64,
        "num_warmup_steps": 4,
        "world_model_window": 3,
        "num_world_model_updates": 1,
        "world_model_batch_size": 8,
        "num_world_model_warmup_iterations": 0,
        "horizon": 3,
        "policy_batch_size": 4,
        "num_critic_epochs": 1,
        "num_critic_mini_batches": 2,
    }
    algorithm.update(options)
    return {
        "algorithm": algorithm,
        "num_steps_per_env": 3,
        "save_interval": 100,
        "multi_gpu": None,
        "obs_groups": {key: [key] for key in ("state", "goal", "context", "labels", "prev_action", "prev_prev_action")},
        "actor": {
            "class_name": "SuperMimicActor",
            "history_length": 2,
            "encoder_hidden_dims": (8,),
            "decoder_hidden_dims": (8,),
        },
        "world_model": {"class_name": "WorldModel", "hidden_dims": (16,), "label_hidden_dims": (8,)},
        "critic": {"class_name": "MLPModel", "hidden_dims": (8,), "obs_normalization": True},
    }


def _collect(learner: SuperMimic, env: ScalarEnv, count: int, inference: bool = True) -> None:
    with torch.inference_mode(inference), torch.no_grad():
        obs = env.get_observations()
        for _ in range(count):
            actions = learner.act(obs)
            obs, reward, done, extras = env.step(actions)
            learner.process_env_step(obs, reward, done, extras)


def _learner(env: ScalarEnv | None = None, **options: object) -> SuperMimic:
    env = ScalarEnv() if env is None else env
    return SuperMimic.construct_algorithm(env.get_observations(), env, _cfg(**options), env.device)


def test_world_model_preserves_state_and_action_gradients_when_weights_are_frozen() -> None:
    """Frozen dynamics weights preserve the Jacobians needed for imagined actor learning."""
    representation = EuclideanStateRepresentation(1)
    model = WorldModel(representation, 1, 1, 1, hidden_dims=(8,))
    with torch.no_grad():
        model.net.output_layer.weight.normal_(0, 0.1)
    model.requires_grad_(False)
    state, action = torch.ones(4, 1, requires_grad=True), torch.ones(4, 1, requires_grad=True)
    model((state,), action, torch.zeros(4, 1))[0].square().sum().backward()
    assert state.grad.abs().sum() > 0 and action.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_direct_and_standard_runner_train_all_models() -> None:
    """The standard loop and direct interface produce finite updates of actor, model and critic."""
    torch.manual_seed(2)
    runner = OffPolicyRunner(ScalarEnv(), _cfg())
    modules = (runner.alg.actor, runner.alg.world_model, runner.alg.critic)
    before = [copy.deepcopy(module.state_dict()) for module in modules]
    runner.learn(3)
    assert runner.alg.iteration > 0
    for old, module in zip(before, modules):
        assert any(not torch.equal(old[name], value) for name, value in module.named_parameters())
    learner, env = _learner(), ScalarEnv()
    _collect(learner, env, 4)
    metrics = learner.update()
    assert {"world_model_loss", "policy_loss", "critic_loss", "label_loss"} <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["policy_grad_norm"] > 0
    for module in (learner.actor, learner.world_model):
        assert all(not buffer.is_inference() for buffer in module.buffers())
    checkpoint = copy.deepcopy(learner.save())
    learner.load(checkpoint, None, strict=True)
    _collect(learner, env, 4, inference=False)
    assert torch.isfinite(torch.tensor(learner.update()["world_model_loss"]))


def test_replay_start_padding_repeats_the_same_noisy_reset_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable history repeats one corrupted frame, as during collection."""
    torch.manual_seed(0)
    env, cfg = ScalarEnv(), _cfg()
    cfg["actor"]["history_length"] = 5
    learner = SuperMimic.construct_algorithm(env.get_observations(), env, cfg, env.device)
    _collect(learner, env, 7)
    index = learner.storage.sample(64, history=4)
    assert (~index.valid).any() and index.valid.all(dim=1).any()
    monkeypatch.setattr(learner.storage, "sample", lambda *args, **kwargs: index)
    monkeypatch.setattr(learner.task, "add_noise", lambda frames: frames + torch.randn_like(frames))
    _, frames = learner._starts(64)
    first = frames[torch.arange(64), index.valid.long().argmax(dim=1)]
    torch.testing.assert_close(frames[~index.valid], first[:, None].expand_as(frames)[~index.valid], rtol=0, atol=0)
    assert (frames[index.valid.all(dim=1), 0] != frames[index.valid.all(dim=1), -1]).any()


def test_inference_device_moves_task_tensors() -> None:
    """Task observation constants move together with policy parameters and history."""
    learner = _learner()
    learner.task.offset = torch.ones(1)
    policy = learner.get_policy().to("meta")
    assert policy.history.device.type == "meta"
    assert learner.task.offset.device.type == "meta"


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_actor_ema_is_passive_over_five_updates(device: str) -> None:
    """Averaging and deployment queries preserve live models, optimizers, actions, losses and RNG exactly."""

    def trajectory(halflife: int | None) -> list[tuple]:
        torch.manual_seed(61)
        env = ScalarEnv(device=device)
        learner = _learner(env, updates_per_step=0.25, actor_ema_halflife=halflife)
        result = []
        for _ in range(5):
            _collect(learner, env, 4)
            metrics = learner.update()
            torch.testing.assert_close(learner.get_policy().output_std, learner.actor.output_std, rtol=0, atol=0)
            saved = copy.deepcopy(learner.save())
            saved.pop("actor_ema_state_dict", None)
            saved.pop("actor_ema_updates", None)
            with torch.inference_mode():
                action = learner.actor.forward_history(learner.actor.history, env.goal).clone()
            result.append((saved, metrics, action, torch.get_rng_state(), torch.cuda.get_rng_state_all()))
        assert learner.iteration == 5
        if halflife is not None:
            assert learner.actor_ema_updates == 5
            assert any(not torch.equal(p, average) for p, average in zip(learner.actor.parameters(), learner.actor_ema))
            for parameter, average in zip(learner.get_policy().parameters(), learner.actor_ema):
                torch.testing.assert_close(parameter, average, rtol=0, atol=0)
        return result

    torch.testing.assert_close(trajectory(None), trajectory(500), rtol=0, atol=0)


def test_actor_ema_matches_closed_form_and_keeps_live_buffers() -> None:
    """Warm-start averaging follows its weighted sum; deployment synchronizes buffers without mutating training."""
    from threading import Lock

    learner = _learner(actor_ema_halflife=1)
    learner.task.handle = Lock()
    with torch.no_grad():
        for average in learner.actor_ema:
            average.zero_()
        for value in range(1, 13):
            for parameter in learner.actor.parameters():
                parameter.fill_(value)
            learner._update_actor_ema(torch.tensor(True))
    decays = [min(0.5, (1 + n) / (10 + n)) for n in range(1, 13)]
    expected = sum((i + 1) * (1 - decay) * math.prod(decays[i + 1 :]) for i, decay in enumerate(decays))
    with torch.inference_mode():
        policy = learner.get_policy()
    assert policy.task is learner.task
    for parameter in policy.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected))
        assert not parameter.is_inference()
    learner.actor.obs_norm.update(torch.randn(4, 2))
    learner.actor.noise_std.fill_(0.3)
    policy = learner.get_policy()
    for target, source in zip(policy.buffers(), learner.actor.buffers()):
        torch.testing.assert_close(target, source, rtol=0, atol=0)
    assert not policy.training and learner.actor.training
    before = copy.deepcopy(learner.save())
    loss = sum(p.sum() for p in learner.actor.parameters()) * float("nan")
    learner._apply(loss, "actor", learner.actor)
    torch.testing.assert_close(learner.save(), before, rtol=0, atol=0)


def test_actor_ema_closed_form_after_optimizer_steps() -> None:
    """The average uses post-step parameters and reaches the half-life decay after warm-up."""
    learner = _learner(actor_ema_halflife=2)
    parameter = next(learner.actor.parameters())
    initial, steps = parameter.detach().double().clone(), []
    for _ in range(30):
        learner._apply(sum(p.sum() for p in learner.actor.parameters()), "actor", learner.actor, clip_value=0.1)
        steps.append(parameter.detach().double().clone())
    decays = [min(0.5**0.5, (1 + n) / (10 + n)) for n in range(1, 31)]
    expected = initial * math.prod(decays) + sum(
        value * (1 - decay) * math.prod(decays[i + 1 :]) for i, (value, decay) in enumerate(zip(steps, decays))
    )
    assert learner.actor_ema_updates == 30
    torch.testing.assert_close(learner.actor_ema[0], expected.float(), rtol=1e-6, atol=1e-7)


def test_actor_ema_checkpoint_full_actor_only_and_legacy_loads() -> None:
    """Full and actor-only loads restore EMA; legacy or disabled loads keep the live policy usable."""
    learner, env = _learner(actor_ema_halflife=500), ScalarEnv()
    _collect(learner, env, 4)
    learner.update()
    saved = copy.deepcopy(learner.save())
    for selection in (None, {"actor": True}):
        restored = _learner(actor_ema_halflife=500)
        restored.load(saved, selection, strict=True)
        assert restored.actor_ema_updates == learner.actor_ema_updates
        torch.testing.assert_close(restored.get_policy().state_dict(), saved["actor_ema_state_dict"], rtol=0, atol=0)
        if selection is None:
            torch.testing.assert_close(restored.save(), saved, rtol=0, atol=0)
        else:
            assert not any(optimizer.state for optimizer in restored.optimizers.values())
            assert restored.iteration == 0
    disabled = _learner()
    with pytest.warns(UserWarning, match="actor EMA; set actor_ema_halflife"):
        disabled.load(saved, {"actor": True}, strict=True)
    torch.testing.assert_close(disabled.get_policy().state_dict(), saved["actor_state_dict"], rtol=0, atol=0)
    assert "actor_ema_state_dict" not in disabled.save()
    legacy = {key: value for key, value in saved.items() if not key.startswith("actor_ema")}
    restored.load(legacy, {"actor": True}, strict=True)
    assert restored.actor_ema_updates == 0
    torch.testing.assert_close(restored.get_policy().state_dict(), legacy["actor_state_dict"], rtol=0, atol=0)
    fresh = _learner(actor_ema_halflife=500)
    fresh.load(legacy, {"actor": True}, strict=True)
    assert fresh.actor_ema_updates == 0
    torch.testing.assert_close(fresh.get_policy().state_dict(), legacy["actor_state_dict"], rtol=0, atol=0)
    malformed = copy.deepcopy(saved)
    name = next(iter(dict(learner.actor.named_parameters())))
    malformed["actor_ema_state_dict"][name] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="size mismatch"):
        restored.load(malformed, {"actor": True}, strict=True)


@pytest.mark.parametrize("halflife", [0, -1])
def test_actor_ema_rejects_nonpositive_halflife(halflife: int) -> None:
    """An enabled EMA requires a positive half-life."""
    with pytest.raises(ValueError, match="actor_ema_halflife"):
        _learner(actor_ema_halflife=halflife)


def test_learning_rate_tracks_learner_iterations_and_resume() -> None:
    """All optimizers share one clamped schedule despite unequal update counts."""
    learner, env = _learner(num_schedule_updates=8, num_world_model_updates=2), ScalarEnv()
    _collect(learner, env, 4)
    learner.update()
    assert learner.iteration == 4
    expected = 2e-4 * (0.03 + 0.97 * 0.5)
    assert all(optimizer.param_groups[0]["lr"] == pytest.approx(expected) for optimizer in learner.optimizers.values())
    restored = _learner(num_schedule_updates=8, num_world_model_updates=2)
    restored.load(copy.deepcopy(learner.save()), None, True)
    _collect(restored, ScalarEnv(), 8)
    restored.update()
    assert all(
        optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 * 0.03) for optimizer in restored.optimizers.values()
    )


def test_missing_final_observations_fail_even_without_dones() -> None:
    """Done-only producers cannot silently train a model on interval pushes."""
    env, learner = ScalarEnv(), _learner()
    learner.act(env.get_observations())
    with pytest.raises(ValueError, match="every-step"):
        learner.process_env_step(env.get_observations(), torch.zeros(4), torch.zeros(4), {})
    with pytest.raises(ValueError, match="every-step"):
        learner.process_env_step(
            env.get_observations(), torch.zeros(4), torch.zeros(4), {"final_obs": env.get_observations()}
        )


def test_interval_edits_cut_only_future_world_model_targets() -> None:
    """Pre-push successors remain unchanged in replay; actor histories continue across pushes."""
    env = ScalarEnv(pushes=True)
    learner = _learner(env)
    _collect(learner, env, 4)
    assert learner.storage.data["window_break"][:4].all()
    window, mask = learner._windows(64)
    assert mask[:, 0].all() and not mask[:, 1:].any()
    assert (window["next_state"][:, 0] < window["state"][:, 0] + 1).all()
    metrics = learner.update()
    assert metrics["window_break_fraction"] == 1.0
    _, history = learner._starts(64)
    assert (history[:, 0, 0] != history[:, -1, 0]).any()


def test_nonzero_crouch_reset_actions_seed_both_predecessors() -> None:
    """Explicit action sets preserve held reset commands instead of assuming zero actions."""
    env = ScalarEnv(reset_action=0.4)
    learner = _learner(env)
    _collect(learner, env, 5)
    torch.testing.assert_close(learner.actor.previous_actions, torch.full((4, 2, 1), 0.4))
    learner.act(env.get_observations())
    torch.testing.assert_close(learner._last_previous_actions, torch.full((4, 2, 1), 0.4))
    assert learner.actor.history[:, -1, -1].eq(0.4).all()


def test_lambda_returns_terminal_reward_and_bootstrap() -> None:
    """A terminal transition retains its reward and excludes future values."""
    actual = lambda_returns(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[0.0], [1.0], [0.0]]),
        torch.tensor([[7.0], [8.0], [9.0], [10.0]]),
        0.5,
        1.0,
    )
    torch.testing.assert_close(actual, torch.tensor([[2.0], [2.0], [8.0]]))


def test_value_cap_bounds_policy_bootstrap_and_critic_targets() -> None:
    """Limit optimistic bootstraps and targets while fitting the uncapped critic."""
    learner, env = _learner(max_step_reward=1.0, gamma=0.5, horizon=1, num_critic_mini_batches=1), ScalarEnv()
    _collect(learner, env, 4)
    with torch.no_grad():
        for parameter in learner.critic.parameters():
            parameter.zero_()
        learner.critic.mlp.output_layer.bias.fill_(10.0)
    metrics = learner._fit_policy()
    assert metrics["policy_loss"].item() == pytest.approx(-metrics["imagined_reward"].item() - 1.0)
    assert metrics["terminal_saturation"].item() == 1.0

    with torch.no_grad():
        learner.critic.mlp.output_layer.bias.fill_(10.0)
    features = torch.zeros(2, 2, learner.critic.obs_dim)
    metrics = learner._fit_critic(features, torch.tensor([[0.0, 4.0]]), torch.zeros(1, 2), torch.ones(1, 2))
    assert metrics["critic_loss"].item() == pytest.approx((9.0**2 + 8.0**2) / 2)
    assert 2.0 < learner.critic.mlp.output_layer.bias.item() < 10.0


@pytest.mark.parametrize(
    ("load_optimizer", "goal_dim", "halflife"), [(False, 0, None), (False, 1, 500), (True, 1, None)]
)
def test_checkpoint_normalized_jit_and_onnx(
    tmp_path: Path, load_optimizer: bool, goal_dim: int, halflife: int | None
) -> None:
    """Full loads, fresh optimizers and stateless deployment preserve normalized policy outputs."""
    from onnx.reference import ReferenceEvaluator

    runner = OffPolicyRunner(ScalarEnv(goal_dim=goal_dim), _cfg(actor_ema_halflife=halflife), str(tmp_path / "logs"))
    runner.learn(2)
    checkpoint = str(tmp_path / "learner.pt")
    runner.save(checkpoint)
    restored = OffPolicyRunner(ScalarEnv(goal_dim=goal_dim), _cfg(actor_ema_halflife=halflife))
    restored.load(checkpoint, load_cfg={**restored.alg.get_default_load_cfg(), "optimizer": load_optimizer})
    assert restored.alg.iteration == runner.alg.iteration
    assert restored.alg.learning_rate == runner.alg.learning_rate
    assert restored.alg.storage.num_steps == 0 and not restored.alg.actor.history.any()
    for key in runner.alg.optimizers:
        assert bool(restored.alg.optimizers[key].state) is load_optimizer
    policy = restored.get_inference_policy()
    history, goal = torch.randn(4, 2, 2), torch.randn(4, goal_dim)
    expected = policy.forward_history(history, goal)
    torch.testing.assert_close(expected, runner.get_inference_policy().forward_history(history, goal), rtol=0, atol=0)
    if halflife is not None:
        assert not torch.equal(expected, restored.alg.actor.forward_history(history, goal))
    restored.export_policy_to_jit(str(tmp_path))
    restored.export_policy_to_onnx(str(tmp_path))
    onnx_model = onnx.load(str(tmp_path / "policy.onnx"))
    onnx.checker.check_model(onnx_model)
    assert [value.name for value in onnx_model.graph.input] == (
        ["obs_history", "goal"] if goal_dim else ["obs_history"]
    )
    exported = torch.jit.load(str(tmp_path / "policy.pt"))
    inputs = (history, goal) if goal_dim else (history,)
    torch.testing.assert_close(exported(*inputs), expected)
    evaluator = ReferenceEvaluator(onnx_model)
    actual = evaluator.run(None, dict(zip(evaluator.input_names, [value[:1].numpy() for value in inputs])))[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected[:1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actor_ema_cpu_inference_survives_exports_from_cuda_runner(tmp_path: Path) -> None:
    """Export and logging queries preserve the device of a returned inference policy."""
    runner = OffPolicyRunner(ScalarEnv(device="cuda:0"), _cfg(actor_ema_halflife=500), device="cuda:0")
    policy = runner.get_inference_policy(device="cpu")
    sample = runner.env.get_observations().cpu()
    with torch.inference_mode():
        runner.export_policy_to_jit(str(tmp_path))
        runner.export_policy_to_onnx(str(tmp_path))
        torch.testing.assert_close(runner.alg.get_policy().output_std, runner.alg.actor.output_std.cpu())
        assert policy(sample).shape == (4, 1)
    assert all(p.device.type == "cpu" and not p.is_inference() for p in policy.parameters())
    assert runner.alg.actor.history.device.type == "cuda"


def test_checkpoint_selection_dimensions_and_strictness() -> None:
    """Actor-only and full model selections validate tensor shapes through state-dict loading."""
    learner = _learner()
    state = copy.deepcopy(learner.save())
    with pytest.raises(ValueError, match="actor-only or full"):
        learner.load(state, {"actor": True, "critic": True}, True)
    bad = dict(state)
    del bad["world_model_state_dict"]
    with pytest.raises(ValueError, match="explicitly contain world_model"):
        learner.load(bad, None, True)
    changed_context = _learner(ScalarEnv(context_dim=2))
    with pytest.raises(RuntimeError, match="size mismatch"):
        changed_context.load(state, None, True)
    changed_context.load(state, {"actor": True}, True)
    changed_goal = _learner(ScalarEnv(goal_dim=2))
    with pytest.raises(RuntimeError, match="size mismatch"):
        changed_goal.load(state, {"actor": True}, True)
    state["actor_state_dict"]["unexpected"] = torch.ones(1)
    with pytest.raises(RuntimeError):
        learner.load(state, {"actor": True}, True)
    learner.load(state, {"actor": True}, False)


def test_failed_policy_update_restores_frozen_parameters() -> None:
    """Task failures cannot leave model/critic parameters frozen."""
    learner, env = _learner(), ScalarEnv()
    _collect(learner, env, 4)

    def fail(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("task failure")

    learner.task.reward = fail
    with pytest.raises(RuntimeError, match="task failure"):
        learner._fit_policy()
    assert all(
        parameter.requires_grad for module in (learner.world_model, learner.critic) for parameter in module.parameters()
    )


def test_fractional_cadence_and_algorithm_level_multi_gpu_rejection() -> None:
    """Direct per-step updates retain the 2/3-step learner cadence without a custom runner hook."""
    learner, env = _learner(updates_per_step=0.4, num_warmup_steps=1, world_model_window=1), ScalarEnv()
    counts = []
    for _ in range(5):
        _collect(learner, env, 1)
        learner.update()
        counts.append(learner.iteration)
    assert counts == [0, 0, 1, 1, 2]
    cfg = _cfg()
    missing_state_cfg = _cfg()
    missing_state_cfg["obs_groups"] = {}
    with pytest.raises(ValueError, match="state"):
        SuperMimic.construct_algorithm(TensorDict({"policy": torch.zeros(4, 1)}, [4]), env, missing_state_cfg, "cpu")
    cfg["multi_gpu"] = {"world_size": 2}
    with pytest.raises(NotImplementedError, match="Multi-GPU"):
        SuperMimic.construct_algorithm(env.get_observations(), env, cfg, "cpu")


def test_task_analytic_successors_follow_labels_in_fitting_and_imagination() -> None:
    """Analytic post-dynamics runs after event prediction and retains the event-probability gradient."""
    learner, env = _learner(), ScalarEnv()
    post_calls = []
    reward = learner.task.reward

    def post(state: State, next_state: State, labels: torch.Tensor | None) -> State:
        successor = (next_state[0] + labels[..., :1],)
        post_calls.append(successor)
        return successor

    def checked_reward(
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert next_state is post_calls[-1]
        return reward(state, actions, next_state, goal, prev_actions, labels)

    learner.task.post_dynamics = post
    learner.task.reward = checked_reward
    state = (torch.ones(4, 1),)
    successor, _, labels = learner._predict(state, torch.zeros(4, 1), torch.zeros(4, 1), torch.zeros(4, 2, 1))
    assert labels is not None
    torch.testing.assert_close(torch.autograd.grad(successor[0].sum(), labels)[0], torch.ones_like(labels))
    _collect(learner, env, 4)
    post_calls.clear()
    learner._fit_world_model()
    assert len(post_calls) == learner.world_model_window
    post_calls.clear()
    assert learner._fit_policy()["policy_grad_norm"] > 0
    assert len(post_calls) == learner.horizon


def test_collection_owns_start_observations() -> None:
    """Replay starts survive an environment reusing its observation tensors during stepping."""
    learner, env = _learner(), ScalarEnv()
    obs = env.get_observations()
    actions = learner.act(obs)
    obs["state"].fill_(100)
    obs["goal"].fill_(200)
    obs["context"].fill_(300)
    successor, rewards, dones, extras = env.step(actions)
    learner.process_env_step(successor, rewards, dones, extras)
    assert learner.storage.data["state"][0].eq(0).all()
    assert learner.storage.data["goal"][0].eq(1).all()
    assert learner.storage.data["context"][0].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
@pytest.mark.parametrize(("mixed_precision", "compile_mode"), [(False, None), (True, None), (True, "default")])
def test_cuda_learning_and_checkpoint_roundtrip(mixed_precision: bool, compile_mode: str | None) -> None:
    """CUDA FP32, BF16 and compiled learning preserve finite model updates and raw checkpoint keys."""
    cfg = _cfg(use_mixed_precision=mixed_precision)
    cfg["torch_compile_mode"] = compile_mode
    torch.manual_seed(2)
    runner = OffPolicyRunner(ScalarEnv(device="cuda"), cfg, device="cuda")
    before = [
        copy.deepcopy(module.state_dict())
        for module in (runner.alg.actor, runner.alg.world_model, runner.alg._raw_critic)
    ]
    runner.learn(2)
    checkpoint = copy.deepcopy(runner.alg.save())
    for old, module in zip(before, (runner.alg.actor, runner.alg.world_model, runner.alg._raw_critic)):
        assert any(not torch.equal(old[name], value) for name, value in module.named_parameters())
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())
    assert not any(
        "_orig_mod" in key for name in ("actor", "critic", "world_model") for key in checkpoint[f"{name}_state_dict"]
    )
    restored = _learner(ScalarEnv(device="cuda"), use_mixed_precision=mixed_precision)
    restored.load(checkpoint, None, True)
    history, goal = torch.randn(4, 2, 2, device="cuda"), torch.randn(4, 1, device="cuda")
    torch.testing.assert_close(
        restored.actor.forward_history(history, goal), runner.alg.actor.forward_history(history, goal)
    )


def test_task_without_successor_event_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tasks without event heads train through the same physical replay and imagination paths."""
    env, cfg = ScalarEnv(), _cfg()
    monkeypatch.setattr(ScalarTask, "num_labels", 0)
    del cfg["obs_groups"]["labels"]
    learner = SuperMimic.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    _collect(learner, env, 4)
    metrics = learner.update()
    assert learner.world_model.label_head is None and "label_loss" not in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["policy_grad_norm"] > 0


@pytest.mark.parametrize("group", ["state", "labels"])
def test_nonfinite_final_observations_are_rejected_before_training(group: str) -> None:
    """Every-step final observations of running environments are consumed, so their NaNs must stop training."""
    env, learner = ScalarEnv(), _learner()
    _collect(learner, env, 4)
    obs = env.get_observations()
    learner.act(obs)
    final = env.get_observations()
    final[group][1] = float("nan")
    learner.process_env_step(obs, torch.zeros(4), torch.zeros(4), {"final_obs": final, "final_obs_every_step": True})
    before = [p.detach().clone() for p in learner.world_model.parameters()]
    with pytest.raises(ValueError, match=f"nonfinite final_obs.*{group}"):
        learner.update()
    assert all(torch.equal(a, b) for a, b in zip(before, learner.world_model.parameters()))


def test_critic_ignores_states_after_imagined_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a certain termination, imagined states neither train the critic nor update its normalizer."""
    env, learner = ScalarEnv(), _learner()
    _collect(learner, env, 4)
    monkeypatch.setattr(learner.task, "termination_prob", lambda *args, **kwargs: torch.ones(4))
    count = int(learner.critic.obs_normalizer.count)
    learner._fit_policy()
    assert int(learner.critic.obs_normalizer.count) - count == learner.policy_batch_size
    features = torch.randn(learner.horizon + 1, 4, learner.critic.obs_normalizer.mean.shape[-1])
    rewards, terminals = torch.ones(learner.horizon, 4), torch.ones(learner.horizon, 4)
    reach = torch.zeros(learner.horizon, 4)
    reach[0] = 1.0
    features[1:] = 1e6
    assert torch.isfinite(torch.tensor(learner._fit_critic(features, rewards, terminals, reach)["critic_loss"].item()))


def test_event_labels_condition_on_the_applied_action() -> None:
    """At the same successor state, different applied actions can lead to different event predictions."""
    learner = _learner()
    with torch.no_grad():
        for parameter in learner.world_model.label_head.parameters():
            parameter.normal_(0, 1.0)
    state = learner.task.representation.unflatten(torch.zeros(4, 1))
    context, previous = torch.zeros(4, 1), torch.zeros(4, learner.task.action_history_length, 1)
    with torch.no_grad():
        _, low, _ = learner._predict(state, torch.full((4, 1), -1.0), context, previous)
        _, high, _ = learner._predict(state, torch.full((4, 1), 1.0), context, previous)
    assert not torch.allclose(low, high)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_cuda_graph_matches_eager_and_resumes() -> None:
    """Graphed learner iterations match eager updates exactly and recapture after loading optimizer state."""

    def train(use_cuda_graph: bool) -> tuple[SuperMimic, ScalarEnv, list[dict[str, float]]]:
        torch.manual_seed(0)
        env = ScalarEnv(device="cuda:0")
        learner = _learner(env, use_cuda_graph=use_cuda_graph, actor_ema_halflife=3)
        metrics = []
        for _ in range(6):
            _collect(learner, env, 3)
            metrics.append(learner.update())
        return learner, env, metrics

    eager, _, eager_metrics = train(False)
    graphed, env, graphed_metrics = train(True)
    assert graphed._graph_updates and graphed_metrics == eager_metrics
    for key in ("actor_state_dict", "critic_state_dict", "world_model_state_dict", "actor_ema_state_dict"):
        torch.testing.assert_close(graphed.save()[key], eager.save()[key], rtol=0, atol=0)
    graphed.load(eager.save(), None, strict=True)
    assert not graphed._graph_updates
    for _ in range(5):
        _collect(graphed, env, 3)
        metrics = graphed.update()
    assert graphed._graph_updates and math.isfinite(metrics["world_model_loss"])
