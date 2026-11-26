# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SHAC learning, replay boundaries and deployment on a synthetic VecEnv."""

from __future__ import annotations

import copy
import gc
import math
import torch
from collections.abc import Iterator
from pathlib import Path
from tensordict import TensorDict

import onnx
import pytest

from rsl_rl.algorithms import SHAC
from rsl_rl.algorithms.shac import compute_lambda_returns
from rsl_rl.env import RolloutState, VecEnv
from rsl_rl.env.differentiable_task import DifferentiableTask, EuclideanStateRepresentation, State, prepare_actor_obs
from rsl_rl.extensions import WorldModel
from rsl_rl.models import MLPModel, SuperMimicActor
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import Muon


class ScalarTask(DifferentiableTask):
    """A Euclidean system with privileged context and one successor event."""

    num_labels = 1

    def __init__(self, env: VecEnv, device: str) -> None:
        """Bind a one-dimensional Euclidean state."""
        self.representation = EuclideanStateRepresentation(1, error_floor=0.01)

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> TensorDict:
        """Expose position, the most recent action and the current goal."""
        frame = torch.cat((state[0], prev_actions[..., -1, :]), -1)
        return TensorDict({"policy": frame, "goal": goal}, batch_size=frame.shape[:-1])

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Expose privileged context and both action predecessors to dynamics."""
        return torch.cat((context, prev_actions.flatten(-2)), -1)

    def critic_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
        history: torch.Tensor,
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
        every_step: bool = True,
        history_length: int = 2,
        reference_actions: bool = False,
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
        self.reset_action, self.pushes, self.every_step = reset_action, pushes, every_step
        self.reference_actions = reference_actions
        self.history = self.position.new_zeros(4, history_length - 1, 2)
        self.reset_history = torch.ones(4, dtype=torch.bool, device=device)

    def get_observations(self) -> TensorDict:
        """Return owned physical observations and reported action predecessors."""
        frame = torch.cat((self.position, self.previous), dim=-1)
        history = torch.where(self.reset_history[:, None, None], frame[:, None], self.history)
        return TensorDict(
            {
                "policy": frame,
                "history": history.flatten(1),
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
        obs = self.get_observations()
        self.history.copy_(torch.cat((obs["history"].reshape_as(self.history), obs["policy"][:, None]), dim=1)[:, 1:])
        if self.reference_actions:
            actions = actions + self.goal
        self.position.add_(0.2 * actions)
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(actions)
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        self.reset_history.zero_()
        final = self.get_observations()
        rewards = 1 - (self.position[:, 0] - 1).square()
        if self.pushes:
            self.position[~dones] += 10
        self.position[dones] = 0
        self.episode_length_buf[dones] = 0
        self.previous[dones] = self.reset_action
        self.previous_previous[dones] = self.reset_action
        self.reset_history.copy_(dones)
        return (
            self.get_observations(),
            rewards,
            dones,
            {"final_obs": final, "final_obs_every_step": self.every_step, "time_outs": dones.clone()},
        )


@pytest.fixture(autouse=True)
def _destroy_finished_cuda_graphs() -> Iterator[None]:
    """Collect learners of finished tests, whose CUDA graphs would invalidate a later capture if destroyed during it.

    Graph-holding learners form reference cycles, and torch.cuda.graph no longer collects garbage before a capture.
    """
    yield
    gc.collect()


def _cfg(**options: object) -> dict:
    dynamics = {
        "model": {"class_name": "WorldModel", "hidden_dims": (16,), "label_hidden_dims": (8,)},
        "replay_buffer_capacity": 64,
        "num_warmup_steps": 4,
        "window_length": 3,
        "num_updates": 1,
        "batch_size": 8,
        "num_warmup_iterations": 0,
        "mirror_fraction": 0.0,
        "full_windows": False,
        "loss_reduction": "sum",
    }
    algorithm = {
        "class_name": "SHAC",
        "history_length": 2,
        "task_cfg": {"class_name": ScalarTask},
        "world_model_cfg": {key: options.pop(key, value) for key, value in dynamics.items()},
        "updates_per_step": 1.0,
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
        "obs_groups": {
            **{key: [key] for key in ("state", "goal", "context", "labels", "prev_action", "prev_prev_action")},
            "actor": ["policy", "goal", "history"],
        },
        "actor": {"class_name": "MLPModel", "hidden_dims": (8,), "obs_normalization": True},
        "critic": {"class_name": "MLPModel", "hidden_dims": (8,), "obs_normalization": True},
    }


def _collect(learner: SHAC, env: ScalarEnv, count: int, inference: bool = True) -> None:
    with torch.inference_mode(inference), torch.no_grad():
        obs = env.get_observations()
        for _ in range(count):
            actions = learner.act(obs)
            obs, reward, done, extras = env.step(actions)
            learner.process_env_step(obs, reward, done, extras)


def _learner(env: ScalarEnv | None = None, **options: object) -> SHAC:
    env = ScalarEnv() if env is None else env
    return SHAC.construct_algorithm(env.get_observations(), env, _cfg(**options), env.device)


def test_world_model_preserves_state_and_action_gradients_when_weights_are_frozen() -> None:
    """Frozen dynamics weights preserve the Jacobians needed for imagined actor learning."""
    representation = EuclideanStateRepresentation(1)
    model = WorldModel(representation, 1, 1, 1, hidden_dims=(8,))
    with torch.no_grad():
        model.dynamics_mlp.output_layer.weight.normal_(0, 0.1)
    model.requires_grad_(False)
    state, action = torch.ones(4, 1, requires_grad=True), torch.ones(4, 1, requires_grad=True)
    model((state,), action, torch.zeros(4, 1))[0].square().sum().backward()
    assert state.grad.abs().sum() > 0 and action.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    representation.integrate = lambda state, increment: ((state[0] + increment) @ torch.eye(1),)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model((state,), action, torch.zeros(4, 1))[0].dtype == torch.float32


def test_direct_and_standard_runner_train_all_models() -> None:
    """The standard loop and direct interface produce finite updates of actor, model and critic."""
    torch.manual_seed(2)
    runner = OnPolicyRunner(ScalarEnv(), _cfg())
    modules = (runner.alg.actor, runner.alg.learned_dynamics.world_model, runner.alg.critic)
    before = [copy.deepcopy(module.state_dict()) for module in modules]
    runner.learn(3)
    assert runner.alg.num_learner_updates > 0
    for old, module in zip(before, modules):
        assert any(not torch.equal(old[name], value) for name, value in module.named_parameters())
    learner, env = _learner(), ScalarEnv()
    _collect(learner, env, 4)
    losses, stats = learner.update()
    assert {"world_model", "policy", "critic", "label"} == losses.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in (losses | stats).values())
    assert stats["SHAC/policy_grad_norm"] > 0 and stats["SHAC/world_model_grad_norm"] > 0
    for module in (learner.actor, learner.learned_dynamics.world_model):
        assert all(not buffer.is_inference() for buffer in module.buffers())
    checkpoint = copy.deepcopy(learner.save())
    learner.load(checkpoint, None, strict=True)
    _collect(learner, env, 4, inference=False)
    assert torch.isfinite(torch.tensor(learner.update()[0]["world_model"]))


def test_policy_update_with_task_dynamics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train and resume SHAC using a differentiable transition supplied by the task."""
    env = ScalarEnv()
    cfg = _cfg(world_model_cfg=None)
    with pytest.raises(ValueError, match="start_rollout and step"):
        SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)

    def start_rollout(task: ScalarTask, batch_size: int) -> RolloutState:
        state = torch.zeros(batch_size, 1)
        actions = torch.zeros(batch_size, 2, 1)
        goal = torch.ones_like(state)
        obs = task.actor_obs((state,), goal, state, actions)
        frames = obs["policy"][:, None].repeat(1, 2, 1)
        return RolloutState((state,), goal, state, actions, frames, obs)

    def step(
        task: ScalarTask, rollout: RolloutState, actions: torch.Tensor, survival: torch.Tensor
    ) -> tuple[RolloutState, torch.Tensor, torch.Tensor]:
        state = (rollout.state[0] + 0.2 * actions,)
        history = torch.cat((rollout.action_history[:, 1:], actions[:, None]), dim=1)
        obs = task.actor_obs(state, rollout.goal, rollout.context, history)
        frames = torch.cat((rollout.obs_history[:, 1:], obs["policy"][:, None]), dim=1)
        next_rollout = RolloutState(
            state,
            rollout.goal,
            rollout.context,
            history,
            frames,
            obs,
        )
        reward = task.reward(rollout.state, actions, state, rollout.goal, rollout.action_history)
        return next_rollout, reward, task.termination_prob(state, rollout.goal)

    monkeypatch.setattr(ScalarTask, "start_rollout", start_rollout)
    monkeypatch.setattr(ScalarTask, "step", step)
    cfg["actor"] = {"class_name": "MLPModel", "hidden_dims": (8,)}
    runner = OnPolicyRunner(env, cfg, device=env.device)
    learner = runner.alg
    assert isinstance(learner.actor, MLPModel)
    obs, rewards, dones, _ = env.step(learner.act(env.get_observations()))
    learner.process_env_step(obs, rewards, dones, {})
    losses, stats = learner.update()
    assert set(losses) == {"policy", "critic"} and stats["SHAC/policy_grad_norm"] > 0
    runner.learn(1)
    saved = learner.save()
    assert "world_model_state_dict" not in saved and set(learner.optimizers) == {"actor", "critic"}
    restored = SHAC.construct_algorithm(obs, env, cfg, env.device)
    assert restored.load(saved, None, strict=True)
    torch.testing.assert_close(restored.save(), saved)


def test_mlp_tracks_a_goal_in_the_policy_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A state-relative target uses the same observations and derivatives with the standard MLP."""

    def waypoint_obs(
        task: ScalarTask, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> TensorDict:
        frame = torch.cat((state[0], goal - state[0]), dim=-1)
        return TensorDict({"policy": frame}, batch_size=frame.shape[:-1])

    monkeypatch.setattr(ScalarTask, "actor_obs", waypoint_obs)
    env, cfg = ScalarEnv(), _cfg()
    env_obs = env.get_observations

    def waypoint_env_obs() -> TensorDict:
        obs = env_obs()
        obs["policy"] = torch.cat((obs["state"], obs["goal"] - obs["state"]), dim=-1)
        return obs

    monkeypatch.setattr(env, "get_observations", waypoint_env_obs)
    cfg["obs_groups"]["actor"] = ["policy"]
    del cfg["algorithm"]["history_length"]
    cfg["actor"] = {"class_name": "MLPModel", "hidden_dims": (8,)}
    learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
    assert learner.actor.obs_groups == ["policy"]
    _collect(learner, env, 4)
    policy = learner.get_policy().eval()
    obs = env.get_observations()
    torch.testing.assert_close(policy(obs), torch.jit.script(policy.as_jit())(obs["policy"]))

    start = learner.learned_dynamics.start_rollout(4)
    action = learner.actor(start.actor_obs)
    successor, _, _ = learner.learned_dynamics.step(start, action, torch.ones(4))
    relative = successor.actor_obs["policy"][..., -1:]
    torch.testing.assert_close(relative, successor.goal - successor.state[0])
    (gradient,) = torch.autograd.grad(relative.sum(), successor.state[0])
    torch.testing.assert_close(gradient, -torch.ones_like(relative))


def test_actor_observation_width_must_match_the_environment() -> None:
    """Catch a rollout history that is shorter than the one exposed to the deployed actor."""
    env = ScalarEnv(history_length=4)
    with pytest.raises(ValueError, match="Imagined actor observation 'history'"):
        SHAC.construct_algorithm(env.get_observations(), env, _cfg(), env.device)


def test_replay_capacity_must_hold_actor_history() -> None:
    """Replay that fits the warmup must also fit the actor's full frame history."""
    env = ScalarEnv(history_length=5)
    learner = _learner(env, history_length=5, num_warmup_steps=4, replay_buffer_capacity=4 * env.num_envs)
    with pytest.raises(ValueError, match="at least 5 steps per environment"):
        _collect(learner, env, 1)


def test_replay_start_padding_repeats_the_same_noisy_reset_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable history repeats one corrupted frame, as during collection."""
    torch.manual_seed(0)
    env, cfg = ScalarEnv(history_length=5), _cfg()
    cfg["algorithm"]["history_length"] = 5
    learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
    _collect(learner, env, 7)
    indices, valid = learner.learned_dynamics.storage._sample_indices(64, history_length=4)
    assert (~valid).any() and valid.all(dim=1).any()
    monkeypatch.setattr(learner.learned_dynamics.storage, "_sample_indices", lambda *args, **kwargs: (indices, valid))
    monkeypatch.setattr(learner.task, "add_noise", lambda frames: frames + torch.randn_like(frames))
    frames = learner.learned_dynamics.start_rollout(64).obs_history
    first = frames[torch.arange(64), valid.long().argmax(dim=1)]
    torch.testing.assert_close(frames[~valid], first[:, None].expand_as(frames)[~valid], rtol=0, atol=0)
    assert (frames[valid.all(dim=1), 0] != frames[valid.all(dim=1), -1]).any()


@pytest.mark.parametrize("halflife", [None, 500])
def test_inference_device_is_independent_of_task_tensors(halflife: int | None) -> None:
    """Deployment uses only the actor, even when task tensors stay on the training device."""
    learner = _learner(actor_ema_halflife=halflife)
    learner.task.offset = torch.ones(1)
    policy = learner.get_policy().to("meta")
    assert isinstance(policy, MLPModel) and next(policy.parameters()).device.type == "meta"
    assert learner.task.offset.device.type == "cpu"


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
        _collect(learner, env, learner.learned_dynamics.num_warmup_steps - 1)
        for _ in range(5):
            _collect(learner, env, 4)
            metrics = learner.update()
            torch.testing.assert_close(
                learner.get_policy().obs_normalizer.mean, learner.actor.obs_normalizer.mean, rtol=0, atol=0
            )
            saved = copy.deepcopy(learner.save())
            saved.pop("actor_ema_state_dict", None)
            saved.pop("actor_ema_updates", None)
            with torch.inference_mode():
                action = learner.actor(env.get_observations()).clone()
            result.append((saved, metrics, action, torch.get_rng_state(), torch.cuda.get_rng_state_all()))
        assert learner.num_learner_updates == 5
        if halflife is not None:
            assert int(learner.actor_ema.num_updates) == 5
            assert any(
                not torch.equal(p, average)
                for p, average in zip(learner.actor.parameters(), learner.actor_ema.averaged_parameters)
            )
            for parameter, average in zip(learner.get_policy().parameters(), learner.actor_ema.averaged_parameters):
                torch.testing.assert_close(parameter, average, rtol=0, atol=0)
        return result

    torch.testing.assert_close(trajectory(None), trajectory(500), rtol=0, atol=0)


def test_actor_ema_matches_closed_form_and_keeps_live_buffers() -> None:
    """Warm-start averaging follows its weighted sum; deployment synchronizes buffers without mutating training."""
    learner = _learner(actor_ema_halflife=1)
    with torch.no_grad():
        for average in learner.actor_ema.averaged_parameters:
            average.zero_()
        for value in range(1, 13):
            for parameter in learner.actor.parameters():
                parameter.fill_(value)
            learner.actor_ema.update_parameters(torch.tensor(True))
    decays = [min(0.5, (1 + n) / (10 + n)) for n in range(1, 13)]
    expected = sum((i + 1) * (1 - decay) * math.prod(decays[i + 1 :]) for i, decay in enumerate(decays))
    with torch.inference_mode():
        policy = learner.get_policy()
    for parameter in policy.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected))
        assert not parameter.is_inference()
    learner.actor.update_normalization(ScalarEnv().get_observations())
    policy = learner.get_policy()
    for target, source in zip(policy.buffers(), learner.actor.buffers()):
        torch.testing.assert_close(target, source, rtol=0, atol=0)
    assert not policy.training and learner.actor.training
    before = copy.deepcopy(learner.save())
    loss = sum(p.sum() for p in learner.actor.parameters()) * float("nan")
    learner._step_model_optimizers(loss, "actor", learner.actor)
    torch.testing.assert_close(learner.save(), before, rtol=0, atol=0)


def test_actor_ema_closed_form_after_optimizer_steps() -> None:
    """The average uses post-step parameters and reaches the half-life decay after warm-up."""
    learner = _learner(actor_ema_halflife=2)
    parameter = next(learner.actor.parameters())
    initial, steps = parameter.detach().double().clone(), []
    for _ in range(30):
        learner._step_model_optimizers(
            sum(p.sum() for p in learner.actor.parameters()), "actor", learner.actor, clip_value=0.1
        )
        steps.append(parameter.detach().double().clone())
    decays = [min(0.5**0.5, (1 + n) / (10 + n)) for n in range(1, 31)]
    expected = initial * math.prod(decays) + sum(
        value * (1 - decay) * math.prod(decays[i + 1 :]) for i, (value, decay) in enumerate(zip(steps, decays))
    )
    assert int(learner.actor_ema.num_updates) == 30
    torch.testing.assert_close(learner.actor_ema.averaged_parameters[0], expected.float(), rtol=1e-6, atol=1e-7)


def test_actor_ema_checkpoint_full_actor_only_and_no_ema_loads() -> None:
    """Full and actor-only loads restore EMA; checkpoints without EMA or disabled EMA keep the live policy usable."""
    learner, env = _learner(actor_ema_halflife=500), ScalarEnv()
    _collect(learner, env, 4)
    learner.update()
    saved = copy.deepcopy(learner.save())
    for selection in (None, {"actor": True}):
        restored = _learner(actor_ema_halflife=500)
        restored.load(saved, selection, strict=True)
        assert int(restored.actor_ema.num_updates) == int(learner.actor_ema.num_updates)
        torch.testing.assert_close(restored.get_policy().state_dict(), saved["actor_ema_state_dict"], rtol=0, atol=0)
        if selection is None:
            torch.testing.assert_close(restored.save(), saved, rtol=0, atol=0)
        else:
            assert not any(optimizer.state for optimizer in restored.optimizers.values())
            assert restored.num_learner_updates == 0
    disabled = _learner()
    with pytest.warns(UserWarning, match="actor EMA; set actor_ema_halflife"):
        disabled.load(saved, {"actor": True}, strict=True)
    torch.testing.assert_close(disabled.get_policy().state_dict(), saved["actor_state_dict"], rtol=0, atol=0)
    assert "actor_ema_state_dict" not in disabled.save()
    without_ema = {key: value for key, value in saved.items() if not key.startswith("actor_ema")}
    restored.load(without_ema, {"actor": True}, strict=True)
    assert int(restored.actor_ema.num_updates) == 0
    torch.testing.assert_close(restored.get_policy().state_dict(), without_ema["actor_state_dict"], rtol=0, atol=0)
    fresh = _learner(actor_ema_halflife=500)
    fresh.load(without_ema, {"actor": True}, strict=True)
    assert int(fresh.actor_ema.num_updates) == 0
    torch.testing.assert_close(fresh.get_policy().state_dict(), without_ema["actor_state_dict"], rtol=0, atol=0)
    malformed = copy.deepcopy(saved)
    name = next(iter(dict(learner.actor.named_parameters())))
    malformed["actor_ema_state_dict"][name] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="size mismatch"):
        restored.load(malformed, {"actor": True}, strict=True)


def test_learning_rate_tracks_learner_iterations_and_resume() -> None:
    """All optimizers share one clamped schedule despite unequal update counts."""
    learner, env = _learner(num_schedule_updates=8, num_updates=2), ScalarEnv()
    _collect(learner, env, learner.learned_dynamics.num_warmup_steps + 3)
    learner.update()
    assert learner.num_learner_updates == 4
    expected = 2e-4 * (0.03 + 0.97 * 0.5)
    assert all(optimizer.param_groups[0]["lr"] == pytest.approx(expected) for optimizer in learner.optimizers.values())
    restored = _learner(num_schedule_updates=8, num_updates=2)
    restored.load(copy.deepcopy(learner.save()), None, True)
    _collect(restored, ScalarEnv(), 8)
    restored.update()
    assert all(
        optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 * 0.03) for optimizer in restored.optimizers.values()
    )


def test_final_observations_are_required_for_dones_and_every_step_capture() -> None:
    """A done or an every-step flag without final observations cannot record the physical successor."""
    env, learner = ScalarEnv(), _learner()
    learner.act(env.get_observations())
    with pytest.raises(ValueError, match="final_obs"):
        learner.process_env_step(env.get_observations(), torch.zeros(4), torch.ones(4), {})
    with pytest.raises(ValueError, match="final_obs"):
        learner.process_env_step(env.get_observations(), torch.zeros(4), torch.zeros(4), {"final_obs_every_step": True})


def test_done_only_final_observations_use_returned_successors() -> None:
    """Without every-step capture, done rows take final observations and pushes stay in the model targets."""
    env = ScalarEnv(pushes=True, every_step=False)
    learner = _learner(env)
    _collect(learner, env, 6)
    data = learner.learned_dynamics.storage.transitions
    assert not data["window_break"][:6].any() and data["dones"][:6].any()
    pushed = ~data["dones"][:4]
    assert (data["next_state"][:4][pushed][:, 0] > data["state"][:4][pushed][:, 0] + 9).all()
    done = data["dones"][:6]
    assert ((data["next_state"][:6][done] - data["state"][:6][done]).abs() < 1).all()


def test_interval_edits_cut_only_future_world_model_targets() -> None:
    """Pre-push successors remain unchanged in replay; actor histories continue across pushes."""
    env = ScalarEnv(pushes=True)
    learner = _learner(env)
    _collect(learner, env, 4)
    assert learner.learned_dynamics.storage.transitions["window_break"][:4].all()
    window, mask = learner.learned_dynamics._sample_world_model_windows(64)
    assert mask[:, 0].all() and not mask[:, 1:].any()
    assert (window["next_state"][:, 0] < window["state"][:, 0] + 1).all()
    torch.testing.assert_close(window["next_state"], window["next_state"][:, :1].expand_as(window["next_state"]))
    _, stats = learner.update()
    assert stats["SHAC/window_break_fraction"] == 1.0
    history = learner.learned_dynamics.start_rollout(64).obs_history
    assert (history[:, 0, 0] != history[:, -1, 0]).any()


def test_environments_keep_separate_episode_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Episode ends and an interval edit of one environment leave the replay and histories of the others intact."""
    torch.manual_seed(0)
    env, cfg = ScalarEnv(reset_action=0.4, history_length=4), _cfg()
    cfg["algorithm"]["history_length"] = 4
    env.episode_length_buf = torch.arange(4)
    learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
    obs, histories = env.get_observations(), []
    for step in range(12):
        actions = learner.act(obs)
        histories.append(torch.cat((obs["history"], obs["policy"]), dim=-1).reshape(4, 4, 2))
        obs, rewards, dones, extras = env.step(actions)
        if step == 6:
            env.position[0] += 10
            obs = env.get_observations()
        learner.process_env_step(obs, rewards, dones, extras)
    data = learner.learned_dynamics.storage.transitions[:12]
    assert data["dones"].any() and data["dones"].sum(dim=1).max() == 1
    assert data["window_break"].nonzero().tolist() == [[6, 0]]
    torch.testing.assert_close(data["next_state"], data["state"] + 0.2 * data["actions"])
    windows, starts = (
        learner.learned_dynamics.storage._sample_indices(1024, length=3),
        learner.learned_dynamics.storage._sample_indices(1024, history_length=3),
    )
    monkeypatch.setattr(
        learner.learned_dynamics.storage,
        "_sample_indices",
        lambda count, length=1, history_length=0: starts if history_length else windows,
    )
    _, mask = learner.learned_dynamics._sample_world_model_windows(1024)
    for slot, expected in ((5, [1.0, 1.0, 0.0]), (6, [1.0, 0.0, 0.0])):
        assert mask[windows[0][:, 0] == 4 * slot].unique(dim=0).tolist() == [expected]
    start = learner.learned_dynamics.start_rollout(1024)
    frames = start.obs_history
    slot, env_id = starts[0][:, -1] // 4, starts[0][:, -1] % 4
    torch.testing.assert_close(start.state[0], data["state"][slot, env_id], rtol=0, atol=0)
    torch.testing.assert_close(frames, torch.stack(histories)[slot, env_id], rtol=0, atol=0)


def test_nonzero_crouch_reset_actions_seed_both_predecessors() -> None:
    """Explicit action sets preserve held reset commands instead of assuming zero actions."""
    env = ScalarEnv(reset_action=0.4)
    learner = _learner(env)
    _collect(learner, env, 5)
    torch.testing.assert_close(learner.action_history, torch.full((4, 2, 1), 0.4))
    learner.act(env.get_observations())
    torch.testing.assert_close(learner._transition["action_history"], torch.full((4, 2, 1), 0.4))


def test_lambda_returns_terminal_reward_and_bootstrap() -> None:
    """A terminal transition retains its reward and excludes future values."""
    actual = compute_lambda_returns(
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, torch.nan]]),
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        torch.tensor([[7.0], [8.0], [9.0], [10.0]]).expand(-1, 2),
        0.5,
        0.5,
    )
    torch.testing.assert_close(actual, torch.tensor([[3.5, 3.5], [2.0, 2.0], [8.0, torch.nan]]), equal_nan=True)


def test_value_cap_bounds_policy_bootstrap_and_critic_targets() -> None:
    """Limit optimistic bootstraps and targets while fitting the uncapped critic."""
    learner, env = _learner(max_step_reward=1.0, gamma=0.5, horizon=1, num_critic_mini_batches=1), ScalarEnv()
    _collect(learner, env, 4)
    with torch.no_grad():
        for parameter in learner.critic.parameters():
            parameter.zero_()
        learner.critic.mlp.output_layer.bias.fill_(10.0)
    metrics = learner._update_actor_and_critic()
    assert metrics["policy_loss"].item() == pytest.approx(-metrics["imagined_reward"].item() - 1.0)
    assert metrics["terminal_saturation"].item() == 1.0

    with torch.no_grad():
        learner.critic.mlp.output_layer.bias.fill_(10.0)
    features = torch.zeros(2, 2, learner.critic.obs_dim)
    metrics = learner._update_critic(features, torch.tensor([[0.0, 4.0]]), torch.zeros(1, 2), torch.ones(1, 2))
    assert metrics["critic_loss"].item() == pytest.approx((9.0**2 + 8.0**2) / 2)
    assert 2.0 < learner.critic.mlp.output_layer.bias.item() < 10.0


def test_critic_targets_use_terminations_lambda_and_reach_weights() -> None:
    """Value targets mix one-step and longer returns, stop at terminations and are weighted by reach."""
    learner = _learner(gamma=0.5, lam=0.5, horizon=2, num_critic_mini_batches=1)
    with torch.no_grad():
        for parameter in learner.critic.parameters():
            parameter.zero_()
        learner.critic.mlp.output_layer.bias.fill_(4.0)
    features = torch.zeros(3, 1, learner.critic.obs_dim)
    rewards, terminals, reach = torch.tensor([[1.0], [2.0]]), torch.full((2, 1), 0.5), torch.tensor([[1.0], [0.5]])
    metrics = learner._update_critic(features, rewards, terminals, reach)
    assert metrics["critic_loss"].item() == pytest.approx((1.0 * (4 - 1.875) ** 2 + 0.5 * (4 - 3.0) ** 2) / 1.5)


def test_imagined_objective_matches_a_step_by_step_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rewards are discounted and weighted by the probability of reaching their step, the final value by survival."""
    torch.manual_seed(0)
    env, learner = ScalarEnv(), _learner(gamma=0.9, horizon=5)
    _collect(learner, env, 8)
    task, actor, seen = learner.task, learner.actor, {}
    with torch.no_grad():
        learner.learned_dynamics.world_model.dynamics_mlp.output_layer.weight.normal_(0, 0.5)
        actor.mlp.output_layer.weight.normal_(0, 0.5)
    monkeypatch.setattr(task, "termination_prob", lambda successor, *args, **kwargs: torch.sigmoid(successor[0][:, 0]))
    monkeypatch.setattr(task, "advance", lambda successor, goal, context: (goal + 0.1 * successor[0], context + 0.05))
    start = learner.learned_dynamics.start_rollout(4)
    frames = start.obs_history
    monkeypatch.setattr(learner.learned_dynamics, "start_rollout", lambda batch_size: start)
    monkeypatch.setattr(learner, "_update_critic", lambda *args: seen.update(reach=args[3]) or {})
    state, goal, context, previous = start.state, start.goal, start.context, start.action_history
    history, reach, objective = frames, [torch.ones(4)], torch.zeros(4)
    for step in range(5):
        action = actor(prepare_actor_obs(task.actor_obs(state, goal, context, previous), history))
        successor, _, labels = learner.learned_dynamics._predict_transition(state, action, context, previous)
        objective = objective + 0.9**step * reach[-1] * task.reward(state, action, successor, goal, previous, labels)
        reach.append(reach[-1] * (1 - task.termination_prob(successor, goal, labels)))
        goal, context = task.advance(successor, goal, context)
        previous = torch.cat((previous[:, 1:], action[:, None]), dim=1)
        history = torch.cat(
            (history[:, 1:], task.actor_obs(successor, goal, context, previous)["policy"][:, None]), dim=1
        )
        state = successor
    objective = objective + 0.9**5 * reach[-1] * learner._predict_values(
        task.critic_obs(state, goal, context, previous, history)
    )
    gradient = torch.autograd.grad(-objective.mean(), list(actor.parameters()))
    metrics = learner._update_actor_and_critic()
    torch.testing.assert_close(metrics["policy_loss"], -objective.mean().detach())
    torch.testing.assert_close(metrics["policy_grad_norm"], torch.cat([g.flatten() for g in gradient]).norm())
    torch.testing.assert_close(seen["reach"], torch.stack(reach[:-1]).detach())


@pytest.mark.parametrize(
    ("load_optimizer", "goal_dim", "halflife"), [(False, 0, None), (False, 1, 500), (True, 1, None)]
)
def test_checkpoint_normalized_jit_and_onnx(
    tmp_path: Path, load_optimizer: bool, goal_dim: int, halflife: int | None
) -> None:
    """Full loads, fresh optimizers and stateless deployment preserve normalized policy outputs."""
    from onnx.reference import ReferenceEvaluator

    runner = OnPolicyRunner(ScalarEnv(goal_dim=goal_dim), _cfg(actor_ema_halflife=halflife), str(tmp_path / "logs"))
    runner.learn(2)
    checkpoint = str(tmp_path / "learner.pt")
    runner.save(checkpoint)
    restored = OnPolicyRunner(ScalarEnv(goal_dim=goal_dim), _cfg(actor_ema_halflife=halflife))
    restored.load(checkpoint, load_cfg={**restored.alg.get_default_load_cfg(), "optimizer": load_optimizer})
    assert restored.alg.num_learner_updates == runner.alg.num_learner_updates
    assert restored.alg.learning_rate == runner.alg.learning_rate
    assert restored.alg.learned_dynamics.storage.num_steps == 0 and not restored.alg.action_history.any()
    for key in runner.alg.optimizers:
        assert bool(restored.alg.optimizers[key].state) is load_optimizer
    policy = restored.get_inference_policy()
    obs = runner.env.get_observations()
    expected = policy(obs)
    torch.testing.assert_close(expected, runner.get_inference_policy()(obs), rtol=0, atol=0)
    if halflife is not None:
        assert not torch.equal(expected, restored.alg.actor(obs))
    restored.export_policy_to_jit(str(tmp_path))
    restored.export_policy_to_onnx(str(tmp_path))
    onnx_model = onnx.load(str(tmp_path / "policy.onnx"))
    onnx.checker.check_model(onnx_model)
    assert [value.name for value in onnx_model.graph.input] == ["obs"]
    inputs = torch.cat([obs[key] for key in policy.obs_groups], dim=-1)
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(inputs), expected)
    evaluator = ReferenceEvaluator(onnx_model)
    actual = evaluator.run(None, {"obs": inputs[:1].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected[:1])


def test_runner_loads_actor_for_inference(tmp_path: Path) -> None:
    """Loading only the actor reproduces the trained policy's actions."""
    runner = OnPolicyRunner(ScalarEnv(), _cfg(), str(tmp_path / "logs"))
    runner.learn(2)
    path = str(tmp_path / "learner.pt")
    runner.save(path)
    restored = OnPolicyRunner(ScalarEnv(), _cfg())
    restored.load(path, load_cfg={"actor": True})
    obs = runner.env.get_observations()
    torch.testing.assert_close(restored.get_inference_policy()(obs), runner.get_inference_policy()(obs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actor_ema_cpu_inference_survives_exports_from_cuda_runner(tmp_path: Path) -> None:
    """Export and logging queries preserve the device of a returned inference policy."""
    runner = OnPolicyRunner(ScalarEnv(device="cuda:0"), _cfg(actor_ema_halflife=500), device="cuda:0")
    policy = runner.get_inference_policy(device="cpu")
    sample = runner.env.get_observations().cpu()
    with torch.inference_mode():
        runner.export_policy_to_jit(str(tmp_path))
        runner.export_policy_to_onnx(str(tmp_path))
        torch.testing.assert_close(
            runner.alg.get_policy().obs_normalizer.mean, runner.alg.actor.obs_normalizer.mean.cpu()
        )
        assert policy(sample).shape == (4, 1)
    assert all(p.device.type == "cpu" and not p.is_inference() for p in policy.parameters())
    assert runner.alg.action_history.device.type == "cuda"


class OffsetScalarTask(ScalarTask):
    """Offset actor observations by a task-owned tensor constant."""

    def __init__(self, env: VecEnv, device: str) -> None:
        """Allocate the offset on the training device."""
        super().__init__(env, device)
        self.offset = torch.zeros(1, device=device)

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> TensorDict:
        """Offset the scalar observation."""
        obs = super().actor_obs(state, goal, context, prev_actions)
        obs["policy"] = obs["policy"] + self.offset
        return obs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_moving_the_actor_ema_policy_leaves_the_training_task_in_place(tmp_path: Path) -> None:
    """A CPU copy of the EMA deployment actor does not move the task of the CUDA learner."""
    cfg = _cfg(actor_ema_halflife=500, task_cfg={"class_name": OffsetScalarTask})
    runner = OnPolicyRunner(ScalarEnv(device="cuda:0"), cfg, str(tmp_path), device="cuda:0")
    runner.learn(2)
    policy = runner.get_inference_policy(device="cpu")
    assert policy(runner.env.get_observations().cpu()).shape == (4, 1)
    assert runner.alg.task.offset.device.type == "cuda"
    runner.learn(2)


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
        learner._update_actor_and_critic()
    assert all(
        parameter.requires_grad
        for module in (learner.learned_dynamics.world_model, learner.critic)
        for parameter in module.parameters()
    )


def test_fractional_cadence_and_distributed_initialization_validation() -> None:
    """Fractional cadence runs a learner iteration every 2-3 steps; missing state and multi-GPU configs fail."""
    learner, env = _learner(updates_per_step=0.4, num_warmup_steps=1, window_length=1), ScalarEnv()
    counts = []
    for _ in range(5):
        _collect(learner, env, 1)
        learner.update()
        counts.append(learner.num_learner_updates)
    assert counts == [0, 0, 1, 1, 2]
    cfg = _cfg()
    missing_state_cfg = _cfg()
    missing_state_cfg["obs_groups"] = {}
    with pytest.raises(ValueError, match="state"):
        SHAC.construct_algorithm(TensorDict({"policy": torch.zeros(4, 1)}, [4]), env, missing_state_cfg, "cpu")
    cfg["multi_gpu"] = {"global_rank": 0, "world_size": 2}
    with pytest.raises(ValueError, match="initialized process group"):
        SHAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")


def test_collection_schedule_keeps_fractional_credit_through_warmup() -> None:
    """The physical-step recipe skips early fits without restarting its cadence or learning-rate clock."""
    learner, env = (
        _learner(
            updates_per_step=0.4,
            update_schedule="from_start",
            full_windows=True,
            num_warmup_steps=4,
            learning_rate_schedule_unit="env_steps",
            num_schedule_updates=10,
            critic_shuffle_each_epoch=False,
        ),
        ScalarEnv(),
    )
    counts = []
    for _ in range(13):
        _collect(learner, env, 1)
        learner.update()
        counts.append(learner.num_learner_updates)
    assert counts == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
    for optimizer in learner.optimizers.values():
        assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4 * 0.03)
    restored = _learner(
        updates_per_step=0.4,
        update_schedule="from_start",
        num_warmup_steps=4,
        learning_rate_schedule_unit="env_steps",
        num_schedule_updates=10,
    )
    restored.load(copy.deepcopy(learner.save()), None, True)
    assert restored.num_env_steps == 13
    resumed_counts = []
    for _ in range(6):
        _collect(restored, env, 1)
        restored.update()
        resumed_counts.append(restored.num_learner_updates)
    assert resumed_counts == [4, 4, 4, 4, 4, 5]


def test_restored_update_credit_matches_the_running_learner() -> None:
    """A load restores the credit of the running learner where the accrued credit rounds below a whole update."""
    options = {"updates_per_step": 0.6, "update_schedule": "from_start"}
    learner, restored, env = _learner(**options), _learner(**options), ScalarEnv()
    for _ in range(8):
        _collect(learner, env, 1)
        learner.update()
        restored.load(copy.deepcopy(learner.save()), None, True)
        assert restored._update_credit == pytest.approx(learner._update_credit, abs=1e-6)


@pytest.mark.parametrize("schedule", ["after_warmup", "from_start"])
def test_warmup_credits_only_steps_that_can_train(schedule: str) -> None:
    """A rollout that crosses warmup only earns updates for eligible steps."""
    env = ScalarEnv()
    learner = _learner(env, update_schedule=schedule)
    _collect(learner, env, learner.learned_dynamics.num_warmup_steps + 1)
    learner.update()
    assert learner.num_learner_updates == 2


def test_collection_can_select_actions_before_updating_moments() -> None:
    """Fresh observations change future normalization without changing the action already being selected."""
    env, cfg = ScalarEnv(), _cfg(action_noise=0.0)
    cfg["algorithm"]["normalize_before_action"] = False
    learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    obs = env.get_observations()
    obs["policy"] += 1.0
    with torch.no_grad():
        expected = learner.actor(obs)
        actual = learner.act(obs)
    torch.testing.assert_close(actual, expected)
    assert learner.actor.obs_normalizer.count == env.num_envs

    torch.testing.assert_close(
        learner.actor.obs_normalizer.mean, torch.cat([obs[k] for k in learner.actor.obs_groups], dim=-1).mean(0)
    )


def test_task_analytic_successors_follow_labels_in_fitting_and_imagination() -> None:
    """Analytic post-dynamics follows event prediction, keeps its gradient and sees the step's action and context."""
    learner, env = _learner(), ScalarEnv()
    post_calls, post_inputs = [], []
    reward = learner.task.reward

    def post(
        state: State, next_state: State, labels: torch.Tensor | None, actions: torch.Tensor, context: torch.Tensor
    ) -> State:
        successor = (next_state[0] + labels[..., :1],)
        post_calls.append(successor)
        post_inputs.append((actions, context))
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
    learner.task.dynamics_actions = lambda actions: 2 * actions
    state, actions, context = (torch.ones(4, 1),), torch.zeros(4, 1), torch.zeros(4, 1)
    successor, logits, labels = learner.learned_dynamics._predict_transition(
        state, actions, context, torch.zeros(4, 2, 1)
    )
    assert torch.equal(labels, torch.sigmoid(logits))
    assert post_inputs[0][0] is actions and post_inputs[0][1] is context
    torch.testing.assert_close(torch.autograd.grad(successor[0].sum(), labels)[0], torch.ones_like(labels))
    _collect(learner, env, 4)
    post_calls.clear()
    learner.learned_dynamics.compute_loss()
    assert len(post_calls) == learner.learned_dynamics.window_length
    post_calls.clear()
    assert learner._update_actor_and_critic()["policy_grad_norm"] > 0
    assert len(post_calls) == learner.num_imagined_steps


def test_task_curriculum_sees_each_collected_step_with_its_pre_step_goal() -> None:
    """The curriculum hook receives the goal the actor acted on, the step's rewards and dones, and the extras."""
    learner, env = _learner(), ScalarEnv()
    calls = []
    learner.task.update_curriculum = lambda *args: calls.append(args)
    obs = env.get_observations()
    actions = learner.act(obs)
    env.goal.fill_(3.0)
    successor, rewards, dones, extras = env.step(actions)
    learner.process_env_step(successor, rewards, dones, extras)
    ((goal, step_rewards, step_dones, step_extras),) = calls
    torch.testing.assert_close(goal, obs["goal"])
    torch.testing.assert_close(step_rewards, rewards)
    assert step_dones.dtype == torch.bool and step_extras is extras


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
    assert learner.learned_dynamics.storage.transitions["state"][0].eq(0).all()
    assert learner.learned_dynamics.storage.transitions["goal"][0].eq(1).all()
    assert learner.learned_dynamics.storage.transitions["context"][0].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
@pytest.mark.parametrize(("mixed_precision", "compile_mode"), [(False, None), (True, None), (True, "default")])
def test_cuda_learning_and_checkpoint_roundtrip(mixed_precision: bool, compile_mode: str | None) -> None:
    """CUDA FP32, BF16 and compiled learning preserve finite model updates and raw checkpoint keys."""
    cfg = _cfg(use_mixed_precision=mixed_precision)
    cfg["torch_compile_mode"] = compile_mode
    torch.manual_seed(2)
    runner = OnPolicyRunner(ScalarEnv(device="cuda"), cfg, device="cuda")
    before = [
        copy.deepcopy(module.state_dict())
        for module in (runner.alg.actor, runner.alg.learned_dynamics.world_model, runner.alg.critic)
    ]
    runner.learn(2)
    checkpoint = copy.deepcopy(runner.alg.save())
    for old, module in zip(before, (runner.alg.actor, runner.alg.learned_dynamics.world_model, runner.alg.critic)):
        assert any(not torch.equal(old[name], value) for name, value in module.named_parameters())
        assert all(torch.isfinite(value).all() for value in module.state_dict().values())
    assert not any(
        "_orig_mod" in key for name in ("actor", "critic", "world_model") for key in checkpoint[f"{name}_state_dict"]
    )
    restored = _learner(ScalarEnv(device="cuda"), use_mixed_precision=mixed_precision)
    restored.load(checkpoint, None, True)
    obs = runner.env.get_observations()
    torch.testing.assert_close(restored.actor(obs), runner.alg.actor(obs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_optimizer_state_resumes_across_devices(use_cuda_graph: bool) -> None:
    """Checkpoints of CPU learners keep training on CUDA with fused Adam, and the reverse."""
    cpu, cpu_env = _learner(), ScalarEnv()
    _collect(cpu, cpu_env, 4)
    cpu.update()
    env = ScalarEnv(device="cuda:0")
    cuda = _learner(env, use_cuda_graph=use_cuda_graph)
    cuda.load(copy.deepcopy(cpu.save()), None, strict=True)
    before = copy.deepcopy(cuda.save())
    for _ in range(5):
        _collect(cuda, env, 3)
        losses, _ = cuda.update()
    assert all(math.isfinite(value) for value in losses.values())
    for name in ("actor_state_dict", "world_model_state_dict"):
        assert any(not torch.equal(before[name][key], value) for key, value in cuda.save()[name].items())
    restored = _learner()
    restored.load(copy.deepcopy(cuda.save()), None, strict=True)
    assert not any(group["fused"] for optimizer in restored.optimizers.values() for group in optimizer.param_groups)
    _collect(restored, cpu_env, 4)
    assert math.isfinite(restored.update()[0]["world_model"])


def test_task_without_successor_event_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tasks without event heads train through the same physical replay and imagination paths."""
    env, cfg = ScalarEnv(), _cfg()
    monkeypatch.setattr(ScalarTask, "num_labels", 0)
    del cfg["obs_groups"]["labels"]
    learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, "cpu")
    _collect(learner, env, 4)
    losses, stats = learner.update()
    assert learner.learned_dynamics.world_model.label_head is None and "label" not in losses
    assert all(torch.isfinite(torch.tensor(value)) for value in (losses | stats).values())
    assert stats["SHAC/policy_grad_norm"] > 0


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
    before = [p.detach().clone() for p in learner.learned_dynamics.world_model.parameters()]
    with pytest.raises(ValueError, match=f"nonfinite final_obs.*{group}"):
        learner.update()
    assert all(torch.equal(a, b) for a, b in zip(before, learner.learned_dynamics.world_model.parameters()))


def test_checkpoint_load_discards_collection_state() -> None:
    """Reloading a checkpoint with finite models clears flags and returns from the discarded rollout."""
    learner = _learner(actor_learning_rate_multiplier=2.0, actor_learning_rate_decay_return=0.5)
    checkpoint = copy.deepcopy(learner.save())
    learner.learned_dynamics.nonfinite_final_obs.fill_(True)
    multiplier = learner.actor_learning_rate_multiplier
    multiplier.episode_returns.fill_(4.0)
    multiplier.completed_return_stats.copy_(torch.tensor([1.0, 1.0]))

    learner.load(checkpoint, None, True)
    learner.update()
    assert multiplier.decay_start_update is None
    assert not multiplier.episode_returns.any()


def test_critic_ignores_states_after_imagined_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a certain termination, imagined states neither train the critic nor update its normalizer."""
    env, learner = ScalarEnv(), _learner()
    _collect(learner, env, 4)
    monkeypatch.setattr(learner.task, "termination_prob", lambda *args, **kwargs: torch.ones(4))
    monkeypatch.setattr(
        learner.learned_dynamics, "_predict_transition", lambda state, *_: ((torch.ones_like(state[0]),), None, None)
    )
    monkeypatch.setattr(
        learner.task,
        "reward",
        lambda state, action, *args, **kwargs: torch.where(state[0][:, 0] == 1, torch.nan, action[:, 0]),
    )
    count = int(learner.critic.obs_normalizer.count)
    metrics = learner._update_actor_and_critic()
    assert torch.isfinite(metrics["policy_loss"]) and torch.isfinite(metrics["policy_grad_norm"])
    assert int(learner.critic.obs_normalizer.count) - count == learner.policy_batch_size
    features = torch.randn(learner.num_imagined_steps + 1, 4, learner.critic.obs_normalizer.mean.shape[-1])
    rewards, terminals = torch.ones(learner.num_imagined_steps, 4), torch.ones(learner.num_imagined_steps, 4)
    reach = torch.zeros(learner.num_imagined_steps, 4)
    reach[0] = 1.0
    features[1:] = torch.nan
    assert torch.isfinite(
        torch.tensor(learner._update_critic(features, rewards, terminals, reach)["critic_loss"].item())
    )


def test_fixed_critic_shuffle_samples_the_entire_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when minibatches drop a remainder, every rollout position is eligible for sampling."""
    learner = _learner(num_critic_epochs=2, critic_shuffle_each_epoch=False)
    monkeypatch.setattr(torch, "randperm", lambda size, device: torch.arange(size - 1, -1, -1, device=device))
    features = torch.arange(12.0).view(4, 3, 1).expand(-1, -1, learner.critic.obs_normalizer.mean.numel())
    value = learner._predict_values
    batches = []

    def record_minibatch(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            batches.append(features[:, 0])
        return value(features)

    monkeypatch.setattr(learner, "_predict_values", record_minibatch)
    learner._update_critic(features, torch.zeros(3, 3), torch.zeros(3, 3), torch.ones(3, 3))
    torch.testing.assert_close(torch.cat(batches), torch.arange(8.0, 0.0, -1).repeat(2))


def test_event_labels_condition_on_the_applied_action() -> None:
    """At the same successor state, different applied actions can lead to different event predictions."""
    learner = _learner()
    with torch.no_grad():
        for parameter in learner.learned_dynamics.world_model.label_head.parameters():
            parameter.normal_(0, 1.0)
    state = learner.task.representation.unflatten(torch.zeros(4, 1))
    context, previous = torch.zeros(4, 1), torch.zeros(4, learner.task.action_history_length, 1)
    with torch.no_grad():
        _, low, _ = learner.learned_dynamics._predict_transition(state, torch.full((4, 1), -1.0), context, previous)
        _, high, _ = learner.learned_dynamics._predict_transition(state, torch.full((4, 1), 1.0), context, previous)
    assert not torch.allclose(low, high)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_cuda_graph_collection_matches_eager_across_resets_and_pushes() -> None:
    """Graphed act and process_env_step store the same replay, histories and statistics as eager collection."""

    def collect(use_cuda_graph: bool) -> tuple[SHAC, list[torch.Tensor]]:
        torch.manual_seed(0)
        env = ScalarEnv(device="cuda:0", reset_action=0.3, pushes=True)
        learner = _learner(env, use_cuda_graph=use_cuda_graph, replay_buffer_capacity=64)
        actions = []
        obs = env.get_observations()
        for _ in range(12):
            actions.append(learner.act(obs))
            obs, reward, done, extras = env.step(actions[-1])
            learner.process_env_step(obs, reward, done, extras)
        return learner, actions

    eager, eager_actions = collect(False)
    graphed, graphed_actions = collect(True)
    graphs = (*graphed._act_graphs.values(), *graphed._process_env_step_graphs.values())
    assert graphs and all(graph.graph is not None for graph in graphs)
    torch.testing.assert_close(graphed_actions, eager_actions, rtol=0, atol=0)
    assert graphed.learned_dynamics.storage.num_steps == eager.learned_dynamics.storage.num_steps == 12
    torch.testing.assert_close(
        graphed.learned_dynamics.storage.transitions[:12].to_dict(),
        eager.learned_dynamics.storage.transitions[:12].to_dict(),
        rtol=0,
        atol=0,
    )
    assert (
        eager.learned_dynamics.storage.transitions["dones"][:12].any()
        and eager.learned_dynamics.storage.transitions["window_break"][:12].any()
    )
    torch.testing.assert_close(graphed.action_history, eager.action_history, rtol=0, atol=0)
    torch.testing.assert_close(
        graphed.learned_dynamics.world_model.state_dict(),
        eager.learned_dynamics.world_model.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        graphed.learned_dynamics._window_break_sum, eager.learned_dynamics._window_break_sum, rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
@pytest.mark.parametrize(("first", "second"), [("train_mode", "eval_mode"), ("eval_mode", "train_mode")])
def test_cuda_graph_collection_follows_mode_switches(first: str, second: str) -> None:
    """Graphed collection updates world-model statistics only in training mode, as eager collection does."""

    def counts(use_cuda_graph: bool) -> list[tuple[int, int]]:
        env = ScalarEnv(device="cuda:0")
        learner = _learner(env, use_cuda_graph=use_cuda_graph)
        result = []
        for mode in (first, second, first):
            getattr(learner, mode)()
            _collect(learner, env, 4)
            result.append((
                int(learner.learned_dynamics.world_model.input_normalizer.count),
                int(learner.learned_dynamics.world_model.increment_normalizer.count),
            ))
        return result

    assert counts(True) == counts(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
@pytest.mark.parametrize(
    "option", ["default", "muon", "actor_multiplier", "reference_action", "mirror", "full_windows", "hooks"]
)
def test_cuda_graph_matches_eager_and_resumes(option: str) -> None:
    """Graphed learner iterations match eager updates exactly with each optional training feature."""
    options = {
        "default": {},
        "muon": {"muon_models": ("actor", "world_model")},
        "actor_multiplier": {"actor_learning_rate_multiplier": 4.0, "actor_learning_rate_decay_return": 0.5},
        "reference_action": {"task_cfg": {"class_name": ReferenceScalarTask}},
        "mirror": {"task_cfg": {"class_name": MirroredScalarTask}, "mirror_fraction": 0.5},
        "full_windows": {"full_windows": True},
        "hooks": {"task_cfg": {"class_name": HookedScalarTask}},
    }[option]

    def train(use_cuda_graph: bool) -> tuple[SHAC, ScalarEnv, list[dict[str, float]]]:
        torch.manual_seed(0)
        env = (
            ScalarEnv(reference_actions=True, device="cuda:0")
            if option == "reference_action"
            else ScalarEnv(device="cuda:0")
        )
        cfg = _cfg(use_cuda_graph=use_cuda_graph, actor_ema_halflife=3, action_noise=0.0, **options)
        cfg["algorithm"]["use_reference_action"] = option == "reference_action"
        if option == "muon":
            # Muon trains the hidden matrices between the first and last layer of each network
            cfg["actor"].update(hidden_dims=(8, 8, 8))
            cfg["algorithm"]["world_model_cfg"]["model"].update(hidden_dims=(8, 8, 8), label_hidden_dims=(8, 8, 8))
        learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
        metrics = []
        for _ in range(6):
            _collect(learner, env, 3)
            metrics.append(learner.update())
        return learner, env, metrics

    eager, _, eager_metrics = train(False)
    graphed, env, graphed_metrics = train(True)
    extra_optimizers = {"actor_muon", "world_model_muon"} if option == "muon" else set()
    assert set(graphed.optimizers) == {"actor", "critic", "world_model"} | extra_optimizers
    assert graphed._update_graphs and graphed_metrics == eager_metrics
    for key in ("actor_state_dict", "critic_state_dict", "world_model_state_dict", "actor_ema_state_dict"):
        torch.testing.assert_close(graphed.save()[key], eager.save()[key], rtol=0, atol=0)
    graphed.load(eager.save(), None, strict=True)
    assert not graphed._update_graphs
    for _ in range(5):
        _collect(graphed, env, 3)
        losses, _ = graphed.update()
    assert graphed._update_graphs and math.isfinite(losses["world_model"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_cuda_graph_of_world_model_warmup_is_released_at_the_first_policy_update() -> None:
    """The captured world-model-only iteration is dropped once policy updates begin."""
    env = ScalarEnv(device="cuda:0")
    learner = _learner(env, use_cuda_graph=True, num_warmup_iterations=5)
    captured = []
    for _ in range(14):
        _collect(learner, env, 1)
        learner.update()
        captured.append({key: graph.graph is not None for key, graph in learner._update_graphs.items()})
    assert {False: True} in captured and captured[-1] == {True: True}


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs torch 2.9 or newer")
def test_muon_matches_torch_and_skips_nonfinite_steps() -> None:
    """The capturable Muon matches PyTorch's Adam-matched Muon and leaves state unchanged on skipped steps."""
    torch.manual_seed(0)
    ours, reference = torch.nn.Parameter(torch.randn(16, 8)), torch.nn.Parameter(torch.randn(16, 8))
    with torch.no_grad():
        reference.copy_(ours)
    muon = Muon([ours], lr=1e-2, weight_decay=0.1)
    torch_muon = torch.optim.Muon([reference], lr=1e-2, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    for _ in range(3):
        ours.grad, reference.grad = (torch.randn(16, 8),) * 2
        muon.step()
        torch_muon.step()
    torch.testing.assert_close(ours, reference)
    before, momentum = ours.detach().clone(), muon.state[ours]["momentum_buffer"].clone()
    muon.found_inf, ours.grad = torch.tensor(1.0), torch.full((16, 8), float("nan"))
    muon.step()
    assert torch.equal(ours.detach(), before) and torch.equal(muon.state[ours]["momentum_buffer"], momentum)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("skip_flag", [False, True])
def test_muon_updates_do_not_depend_on_the_learning_rate_type(device: str, skip_flag: bool) -> None:
    """CUDA graphs hold learning rates as float32 tensors, which must update exactly like eager float rates."""
    torch.manual_seed(0)
    grad = torch.randn(8, 8, device=device)
    updated = []
    for lr in (2e-4, torch.tensor(2e-4, device=device)):
        param = torch.nn.Parameter(torch.zeros(8, 8, device=device))
        muon = Muon([param], lr=lr, weight_decay=0.1)
        muon.found_inf = torch.zeros((), device=device) if skip_flag else None
        param.grad = grad.clone()
        muon.step()
        updated.append(param.detach())
    assert torch.equal(*updated)


def test_muon_trains_hidden_matrices_and_resumes() -> None:
    """Muon owns the inner MLP weights of the actor and world model; the critic and other parameters stay on Adam."""
    env = ScalarEnv()
    cfg = _cfg(muon_models=("actor", "world_model"))
    cfg["actor"]["hidden_dims"] = (8, 8)
    cfg["algorithm"]["world_model_cfg"]["model"].update(hidden_dims=(16, 16), label_hidden_dims=(8, 8))
    learner = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
    assert set(learner.optimizers) == {"actor", "critic", "world_model", "actor_muon", "world_model_muon"}
    muon = {id(p) for group in learner.optimizers["world_model_muon"].param_groups for p in group["params"]}
    assert muon == {
        id(learner.learned_dynamics.world_model.dynamics_mlp[3].weight),
        id(learner.learned_dynamics.world_model.label_head[3].weight),
    }
    adam = {id(p) for group in learner.optimizers["world_model"].param_groups for p in group["params"]}
    assert not muon & adam and len(muon | adam) == len(list(learner.learned_dynamics.world_model.parameters()))
    _collect(learner, env, 8)
    assert math.isfinite(learner.update()[0]["world_model"])
    saved = learner.save()
    assert {"world_model_muon_optimizer_state_dict", "actor_muon_optimizer_state_dict"} <= saved.keys()
    restored = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
    restored.load(saved, None, strict=True)
    torch.testing.assert_close(restored.save(), saved, rtol=0, atol=0)
    cfg["algorithm"]["muon_models"] = ("world_model",)
    world_model_only = SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)
    assert set(world_model_only.optimizers) == {"actor", "critic", "world_model", "world_model_muon"}
    with pytest.raises(ValueError, match="muon_models"):
        _learner(muon_models=("critic",))


def test_actor_learning_rate_follows_the_shared_schedule() -> None:
    """A separate actor learning rate keeps its ratio to the shared rate through the cosine schedule."""
    learner = _learner(actor_learning_rate=8e-4, num_schedule_updates=10)
    for _ in range(4):
        learner.num_learner_updates += 1
        learner._set_learning_rates()
        actor_lr = learner.optimizers["actor"].param_groups[0]["lr"]
        assert actor_lr == pytest.approx(4 * learner.optimizers["critic"].param_groups[0]["lr"])
        assert learner.learning_rate == learner.optimizers["world_model"].param_groups[0]["lr"]


def _actor_ratio(learner: SHAC) -> float:
    return learner.optimizers["actor"].param_groups[0]["lr"] / learner.optimizers["critic"].param_groups[0]["lr"]


def test_actor_learning_rate_multiplier_decays_after_return_or_iteration() -> None:
    """The multiplier holds until episodes reach the decay return or start iteration, then decays linearly to 1."""
    env = ScalarEnv()
    learner = _learner(
        env,
        actor_learning_rate_multiplier=4.0,
        actor_learning_rate_decay_iterations=2,
        actor_learning_rate_decay_start=50,
    )
    assert _actor_ratio(learner) == pytest.approx(4.0)
    learner.num_learner_updates = 51
    learner._set_learning_rates()
    assert learner.actor_learning_rate_multiplier.decay_start_update == 50 and _actor_ratio(learner) == pytest.approx(
        2.5
    )
    learner.num_learner_updates = 60
    learner._set_learning_rates()
    assert _actor_ratio(learner) == pytest.approx(1.0)

    learner = _learner(env, actor_learning_rate_multiplier=4.0, actor_learning_rate_decay_return=float("-inf"))
    _collect(learner, env, 6)
    learner.update()
    start = learner.actor_learning_rate_multiplier.decay_start_update
    assert start is not None and start <= learner.num_learner_updates
    learner.num_learner_updates = start + 1
    learner._set_learning_rates()
    assert _actor_ratio(learner) == pytest.approx(4.0 + (1.0 - 4.0) / 300)
    restored = _learner(env, actor_learning_rate_multiplier=4.0, actor_learning_rate_decay_return=float("-inf"))
    restored.load(learner.save(), None, strict=True)
    assert restored.actor_learning_rate_multiplier.decay_start_update == start

    never = _learner(env, actor_learning_rate_multiplier=4.0, actor_learning_rate_decay_return=float("inf"))
    _collect(never, env, 6)
    _, stats = never.update()
    assert never.actor_learning_rate_multiplier.decay_start_update is None and _actor_ratio(never) == pytest.approx(4.0)
    assert stats["Optimizer/actor_learning_rate"] == pytest.approx(4.0 * stats["Optimizer/learning_rate"])
    with pytest.raises(ValueError, match="decay"):
        _learner(env, actor_learning_rate_multiplier=4.0)


class FixedEpisodeEnv(ScalarEnv):
    """Episodes end after a fixed number of steps, with reward 1 per step, below a longer maximum length."""

    def __init__(self, episode_steps: int, max_episode_length: int | torch.Tensor) -> None:
        """Allocate the scalar system with the given episode and maximum lengths."""
        super().__init__()
        self.episode_steps, self.max_episode_length = episode_steps, max_episode_length

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """End episodes after ``episode_steps`` and reward every step equally."""
        limit, self.max_episode_length = self.max_episode_length, self.episode_steps
        obs, rewards, dones, extras = super().step(actions)
        self.max_episode_length = limit
        return obs, torch.ones_like(rewards), dones, extras


@pytest.mark.parametrize("per_env", [False, True])
@pytest.mark.parametrize(("episode_steps", "starts"), [(2, False), (20, True)])
def test_actor_learning_rate_decay_return_counts_short_episodes_low(
    episode_steps: int, starts: bool, per_env: bool
) -> None:
    """Returns are divided by the maximum episode length, so short episodes with high reward do not start the decay."""
    env = FixedEpisodeEnv(episode_steps, max_episode_length=torch.full((4,), 20) if per_env else 20)
    learner = _learner(
        env, actor_learning_rate_multiplier=4.0, actor_learning_rate_decay_return=0.5, replay_buffer_capacity=64
    )
    _collect(learner, env, 24)
    learner.update()
    assert (learner.actor_learning_rate_multiplier.decay_start_update is not None) is starts


def test_per_environment_episode_lengths_are_only_read_for_the_decay_return() -> None:
    """VecEnv allows one maximum episode length per environment; the default recipe does not read it."""
    env = ScalarEnv()
    env.max_episode_length = torch.full((4,), 5)
    learner = _learner(env)
    _collect(learner, env, 4)
    assert math.isfinite(learner.update()[0]["world_model"])


def test_actor_learning_rate_decay_return_requires_max_episode_length() -> None:
    """Without the environment's maximum episode length, returns cannot be normalized."""
    env = ScalarEnv()
    del env.max_episode_length
    with pytest.raises(ValueError, match="max_episode_length"):
        _learner(env, actor_learning_rate_multiplier=4.0, actor_learning_rate_decay_return=0.5)


class MirroredScalarTask(ScalarTask):
    """The scalar system is symmetric under negation of all physical quantities."""

    def mirror_state(self, state: torch.Tensor) -> torch.Tensor:
        """Negate the position."""
        return -state

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Negate the actions."""
        return -actions

    def mirror_context(self, context: torch.Tensor) -> torch.Tensor:
        """Negate the context."""
        return -context

    def mirror_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Keep the event labels."""
        return labels


def test_world_model_mirror_augmentation_replaces_a_fixed_share_of_windows() -> None:
    """Mirror the first windows of each batch and preserve the remaining samples and validity."""
    env = ScalarEnv()
    learner = _learner(env, mirror_fraction=0.25, task_cfg={"class_name": MirroredScalarTask})
    assert learner.learned_dynamics.num_mirrored_windows == 2
    _collect(learner, env, 8)
    torch.manual_seed(0)
    mirrored, mirrored_mask = learner.learned_dynamics._sample_world_model_windows(8)
    learner.learned_dynamics.num_mirrored_windows = 0
    torch.manual_seed(0)
    window, mask = learner.learned_dynamics._sample_world_model_windows(8)
    for key in ("state", "next_state", "actions", "action_history", "context"):
        torch.testing.assert_close(mirrored[key][:2], -window[key][:2])
    torch.testing.assert_close(mirrored["labels"], window["labels"])
    torch.testing.assert_close(mirrored[2:], window[2:])
    assert torch.equal(mirrored_mask, mask)
    with pytest.raises(ValueError, match="mirror_state, mirror_actions, mirror_context, mirror_labels"):
        _learner(env, mirror_fraction=0.25)


class ReferenceScalarTask(ScalarTask):
    """Track a reference position by correcting a reference action equal to the goal."""

    def reference_action(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Use the first goal coordinate as the reference action."""
        return goal[..., :1]


def test_reference_action_offsets_collection_imagination_and_exports(tmp_path: Path) -> None:
    """Replay stores the applied command, while the exported MLP returns the environment's correction."""
    env = ScalarEnv(reference_actions=True)
    cfg = _cfg(task_cfg={"class_name": ReferenceScalarTask}, use_reference_action=True, action_noise=0.0)
    runner = OnPolicyRunner(env, cfg, str(tmp_path / "logs"))
    learner = runner.alg
    learner.eval_mode()
    obs = env.get_observations()
    correction = learner.act(obs)
    torch.testing.assert_close(correction, learner.get_policy()(obs))
    next_obs, reward, done, extras = env.step(correction)
    learner.process_env_step(next_obs, reward, done, extras)
    applied = correction + obs["goal"]
    torch.testing.assert_close(learner.learned_dynamics.storage.transitions["actions"][0], applied)
    torch.testing.assert_close(next_obs["state"], obs["state"] + 0.2 * applied)
    runner.export_policy_to_jit(str(tmp_path))
    inputs = torch.cat([obs[key] for key in learner.actor.obs_groups], dim=-1)
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(inputs), correction)
    runner.learn(2)
    assert learner.num_learner_updates > 0


def test_supermimic_actor_uses_shared_frame_statistics_and_current_goal() -> None:
    """Collection moments and imagined input gradients follow the original two-branch policy."""
    torch.manual_seed(3)
    history = torch.randn(4, 5, 2, requires_grad=True)
    goal = torch.randn(4, 3, requires_grad=True)
    obs = TensorDict({"history": history[:, :-1].flatten(1), "policy": history[:, -1], "goal": goal}, [4])
    actor = SuperMimicActor(
        obs, {"actor": ["policy", "goal", "history"]}, "actor", 2, encoder_hidden_dims=(8,), decoder_hidden_dims=(8,)
    )
    actor.update_normalization(obs)
    policy = actor.policy
    assert policy.obs_norm.count == policy.goal_norm.count == 4
    torch.testing.assert_close(policy.obs_norm.mean, history[:, -1].mean(0))
    torch.testing.assert_close(policy.goal_norm.mean, goal.mean(0))
    latent = policy.encoder(policy.goal_norm(goal)).tanh()
    expected = policy.decoder(torch.cat((latent, policy.obs_norm(history).flatten(1)), dim=-1))
    actual = actor(obs)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual.sum(), (history, goal)), torch.autograd.grad(expected.sum(), (history, goal))
    ):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("with_goal", [False, True])
@pytest.mark.parametrize("with_reference", [False, True])
def test_supermimic_actor_training_and_exports(tmp_path: Path, with_goal: bool, with_reference: bool) -> None:
    """Export the trained policy with its original input signature and optional additive reference."""
    from onnx.reference import ReferenceEvaluator

    env = ScalarEnv(reference_actions=with_reference)
    cfg = _cfg(task_cfg={"class_name": ReferenceScalarTask}, use_reference_action=with_reference, actor_ema_halflife=10)
    cfg["actor"] = {"class_name": "SuperMimicActor", "encoder_hidden_dims": (8,), "decoder_hidden_dims": (8,)}
    if not with_goal:
        cfg["obs_groups"]["actor"].remove("goal")
    runner = OnPolicyRunner(env, cfg)
    runner.learn(2)
    assert runner.alg.actor_ema.num_updates > 0
    obs = env.get_observations()
    expected = runner.get_inference_policy()(obs)
    inputs = {"obs_history": torch.cat((obs["history"], obs["policy"]), dim=-1).reshape(env.num_envs, 2, -1)}
    if with_goal:
        inputs["goal"] = obs["goal"]
    if with_reference:
        inputs["reference_action"] = obs["goal"]
        expected = expected + inputs["reference_action"]

    runner.export_policy_to_jit(str(tmp_path))
    runner.export_policy_to_onnx(str(tmp_path))
    scripted = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(scripted(**inputs), expected)
    exported = onnx.load(str(tmp_path / "policy.onnx"))
    assert [value.name for value in exported.graph.input] == list(inputs)
    actual = ReferenceEvaluator(exported).run(None, {key: value[:1].numpy() for key, value in inputs.items()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected[:1])


def test_reference_action_is_ignored_unless_enabled() -> None:
    """Disabling reference corrections passes the full policy action to replay."""
    env = ScalarEnv()
    learner = _learner(env, task_cfg={"class_name": ReferenceScalarTask})
    action = learner.act(env.get_observations())
    torch.testing.assert_close(action, learner._transition["actions"])


def test_reference_action_requires_a_task_reference_at_construction() -> None:
    """A learner that corrects a reference action fails to build when the task has no reference action."""
    cfg = _cfg()
    cfg["algorithm"]["use_reference_action"] = True
    env = ScalarEnv()
    with pytest.raises(ValueError, match="reference_action"):
        SHAC.construct_algorithm(env.get_observations(), env, cfg, env.device)


def test_dynamics_actions_may_change_the_action_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """The world model is sized for the physical command, not for the policy output."""
    monkeypatch.setattr(ScalarTask, "dynamics_actions", lambda self, actions: torch.cat((actions, -actions), dim=-1))
    env, learner = ScalarEnv(), _learner()
    _collect(learner, env, 4)
    model = learner.learned_dynamics.world_model
    assert model._context_start == model.feature_dim + 2
    assert math.isfinite(learner.update()[0]["policy"])


def test_task_model_loss_uses_integrated_coordinates_and_valid_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """A task may supervise coordinates beyond the predicted increment without window-length scaling."""
    env, learner = ScalarEnv(), _learner(loss_reduction="mean")
    _collect(learner, env, 4)
    learner.task.num_labels = 0

    def integrated_loss(prediction: State, target: State, scale: torch.Tensor) -> torch.Tensor:
        coordinate = prediction[0][..., 0]
        return coordinate * 0 + 2.75

    monkeypatch.setattr(learner.task, "world_model_loss", integrated_loss)
    _, result = learner.learned_dynamics.compute_loss()
    assert result["world_model_loss"].item() == pytest.approx(2.75)
    assert result["world_model_error"].item() == pytest.approx(2.75)


def test_world_model_loss_sums_scaled_rollout_errors_over_valid_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each window is rolled out from its first state and adds its valid errors in units of the increment scale."""
    torch.manual_seed(0)
    env, learner = ScalarEnv(), _learner(action_noise=1.0)
    _collect(learner, env, 6)
    window, mask = learner.learned_dynamics._sample_world_model_windows(64)
    assert not mask.all() and mask[:, 1:].any()
    monkeypatch.setattr(learner.learned_dynamics, "_sample_world_model_windows", lambda *_: (window, mask))
    with torch.no_grad():
        learner.learned_dynamics.world_model.label_head.output_layer.weight.zero_()
        learner.learned_dynamics.world_model.label_head.output_layer.bias.fill_(1.0)
    entropy = math.log1p(math.e) - window["labels"][..., 0]
    data = learner.learned_dynamics.storage.transitions[:6]
    increments = data["next_state"] - data["state"]
    rollout = window["state"][:, :1] + increments.mean() * torch.arange(1, 4)[:, None]
    for floor in (0.01, 1.0):
        learner.learned_dynamics.error_floor.fill_(floor)
        errors = ((rollout - window["next_state"]) / increments.std(unbiased=False).clamp(min=floor)).square()[..., 0]
        _, metrics = learner.learned_dynamics.compute_loss()
        assert metrics["label_loss"].item() == pytest.approx(((entropy * mask).sum() / mask.sum()).item())
        expected = ((errors * mask).sum(dim=1).mean() + metrics["label_loss"]) / learner.learned_dynamics.window_length
        assert metrics["world_model_loss"].item() == pytest.approx(expected.item(), rel=1e-4)


def test_raw_world_model_preserves_task_normalized_context() -> None:
    """Physical residual outputs and fixed-range context bypass empirical rescaling when requested."""
    model = WorldModel(
        EuclideanStateRepresentation(1),
        1,
        1,
        1,
        hidden_dims=(8,),
        activation="gelu",
        normalize_output=False,
        normalize_context=False,
        zero_initialize_output=False,
    )
    state, action, context = torch.ones(4, 1), torch.ones(4, 1) * 2, torch.ones(4, 1) * 0.75
    model.update_normalization((state,), action, context, (state + 7,))
    seen = []
    handle = model.dynamics_mlp.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach()))
    with torch.no_grad():
        model.dynamics_mlp.output_layer.weight.zero_()
        model.dynamics_mlp.output_layer.bias.fill_(3.0)
    result = model((state,), action, context)[0]
    handle.remove()
    torch.testing.assert_close(result, state + 3)
    torch.testing.assert_close(seen[0][:, -1:], context)
    torch.testing.assert_close(seen[0][:, :2], torch.zeros(4, 2))


def test_default_world_model_normalizes_inputs_and_rescales_increments() -> None:
    """Dynamics and event inputs use the moments of the observed inputs; outputs are scaled by increment moments."""
    model = WorldModel(EuclideanStateRepresentation(1), 1, 1, 1, hidden_dims=(8,), num_labels=1, label_hidden_dims=(8,))
    state, action, context = torch.tensor([[0.0], [2.0]]), torch.tensor([[1.0], [3.0]]), torch.tensor([[10.0], [30.0]])
    model.update_normalization((state,), action, context, (state + torch.tensor([[1.0], [5.0]]),))
    inputs = torch.cat((state, action, context), -1)
    normalized = (inputs - inputs.mean(0)) / (inputs.std(0, unbiased=False) + model.input_normalizer.eps)
    seen = []
    model.dynamics_mlp.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    model.label_head.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    with torch.no_grad():
        model.dynamics_mlp.output_layer.bias.fill_(2.0)
        result = model((state,), action, context)[0]
        model.predict_label_logits((state,), context)
    torch.testing.assert_close(seen[0], normalized)
    torch.testing.assert_close(seen[1], normalized[:, [0, 2]])
    torch.testing.assert_close(result, state + 3 + 2 * (2 + model.increment_normalizer.eps))


class HistoryCriticScalarTask(ScalarTask):
    """The critic also reads the actor's observation history."""

    def critic_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """Append the flattened actor history to the scalar-value inputs."""
        return torch.cat((super().critic_obs(state, goal, context, prev_actions, history), history.flatten(-2)), -1)


def test_critic_reads_the_actor_history_of_each_imagined_state() -> None:
    """Critic widths include the history, and each imagined state is valued with the history the actor acts on."""
    env = ScalarEnv()
    learner = _learner(env, task_cfg={"class_name": HistoryCriticScalarTask})
    assert (
        learner.critic.obs_dim
        == 5 + learner.learned_dynamics.history_length * env.get_observations()["policy"].shape[-1]
    )
    _collect(learner, env, 4)
    critic_histories, actor_histories = [], []
    critic_obs, actor_forward = learner.task.critic_obs, learner._actor_forward

    def recorded_critic_obs(*args: torch.Tensor) -> torch.Tensor:
        assert torch.equal(args[-1][:, -1], learner.task.actor_obs(*args[:-1])["policy"])
        critic_histories.append(args[-1])
        return critic_obs(*args)

    def recorded_actor(obs: TensorDict) -> torch.Tensor:
        actor_histories.append(torch.cat((obs["history"], obs["policy"]), dim=-1).reshape_as(critic_histories[-1]))
        return actor_forward(obs)

    learner.task.critic_obs, learner._actor_forward = recorded_critic_obs, recorded_actor
    learner._update_actor_and_critic()
    assert len(critic_histories) == len(actor_histories) + 1 == learner.num_imagined_steps + 1
    assert all(torch.equal(critic, actor) for critic, actor in zip(critic_histories, actor_histories))
    assert torch.equal(critic_histories[-1][:, :-1], actor_histories[-1][:, 1:])
    assert critic_histories[-1].requires_grad


class LoadScalarTask(ScalarTask):
    """The successor label is a regressed load instead of an event probability."""

    def label_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Read the load prediction unchanged."""
        return logits

    def label_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Regress loads with a squared error."""
        return (logits - labels).square().mean(dim=-1)


def test_task_label_hooks_define_label_values_and_their_masked_loss() -> None:
    """Imagination reads the task's label activation and the model fit averages its loss over valid positions."""
    torch.manual_seed(0)
    env = ScalarEnv()
    learner = _learner(env, task_cfg={"class_name": LoadScalarTask})
    state = learner.task.representation.unflatten(torch.ones(4, 1))
    _, logits, labels = learner.learned_dynamics._predict_transition(
        state, torch.ones(4, 1), torch.zeros(4, 1), torch.zeros(4, 2, 1)
    )
    torch.testing.assert_close(labels, logits, rtol=0, atol=0)
    _collect(learner, env, 4)
    window, mask = learner.learned_dynamics._sample_world_model_windows(64)
    assert not mask.all()
    losses = []
    label_loss = learner.task.label_loss

    def recorded_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses.append(label_loss(logits, labels))
        return losses[-1]

    learner.learned_dynamics._sample_world_model_windows = lambda *_: (window, mask)
    learner.task.label_loss = recorded_label_loss
    _, metrics = learner.learned_dynamics.compute_loss()
    expected = (losses[0] * mask).sum() / mask.sum()
    torch.testing.assert_close(metrics["label_loss"], expected.detach())


def test_model_losses_must_return_a_loss_per_replay_position() -> None:
    """A label or dynamics loss reduced over replay positions fails the model fit."""
    env = ScalarEnv()
    learner = _learner(env, task_cfg={"class_name": LoadScalarTask})
    _collect(learner, env, 4)
    learner.task.label_loss = lambda logits, labels: (logits - labels).square().mean()
    with pytest.raises(ValueError, match="label_loss must return one value per replay position"):
        learner.learned_dynamics.compute_loss()
    learner.task.world_model_loss = lambda prediction, target, scale: torch.zeros((), requires_grad=True)
    with pytest.raises(ValueError, match="world_model_loss must return one value per replay position"):
        learner.learned_dynamics.compute_loss()


class ImpulseScalarTask(ReferenceScalarTask):
    """A step-counting context schedules a position impulse that acts after the actor has chosen its action."""

    def pre_dynamics(self, state: State, context: torch.Tensor) -> State:
        """Add the scheduled impulse to the transition start."""
        return (state[0] + context,)

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Also expose the transition start to dynamics."""
        return torch.cat((state[0], super().dynamics_context(state, context, prev_actions)), -1)

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Count imagined steps in the context."""
        return goal, context + 1


class CountingScalarEnv(ScalarEnv):
    """The context counts collected steps."""

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Count the step, then advance the system."""
        self.context.add_(1.0)
        return super().step(actions)


def test_pre_dynamics_edits_the_start_of_statistics_fitted_and_imagined_transitions() -> None:
    """Each impulse moves the start seen by the model, post-dynamics and reward, not by replay, actor or critic."""
    env = CountingScalarEnv(reference_actions=True)
    learner = _learner(env, task_cfg={"class_name": ImpulseScalarTask}, use_reference_action=True)
    _collect(learner, env, 4)
    data = learner.learned_dynamics.storage.transitions[:4]
    assert torch.equal(data["state"][1:], data["next_state"][:-1])
    for column in (0, 2):
        torch.testing.assert_close(
            learner.learned_dynamics.world_model.input_normalizer.mean[column], (data["state"] + data["context"]).mean()
        )
    window, mask = learner.learned_dynamics._sample_world_model_windows(learner.learned_dynamics.batch_size)
    start = learner.learned_dynamics.start_rollout(learner.policy_batch_size)
    learner.learned_dynamics._sample_world_model_windows, learner.learned_dynamics.start_rollout = (
        lambda *_: (window, mask),
        lambda batch_size: start,
    )
    seen = {"critic_obs": [], "actor_obs": [], "reference_action": [], "post_dynamics": [], "reward": []}

    def record(name: str) -> None:
        hook = getattr(learner.task, name)

        def recorded(state: State, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor | State:
            seen[name].append((state, hook(state, *args, **kwargs)))
            return seen[name][-1][1]

        setattr(learner.task, name, recorded)

    for name in seen:
        record(name)
    learner.learned_dynamics.compute_loss()
    learner._update_actor_and_critic()
    steps, horizon = learner.learned_dynamics.window_length, learner.num_imagined_steps
    fitted, imagined = seen["post_dynamics"][:steps], seen["post_dynamics"][steps:]
    observed = [window["state"][:, 0], *(successor[0] for _, successor in fitted[:-1])]
    observed += [state[0] for state, _ in seen["critic_obs"][:-1]]
    contexts = [window["context"][:, step] for step in range(steps)]
    contexts += [start.context + step for step in range(horizon)]
    assert len(observed) == len(contexts) == len(seen["post_dynamics"]) == steps + horizon
    for (edited, _), state, context in zip(seen["post_dynamics"], observed, contexts):
        assert torch.equal(edited[0], state + context)
    assert torch.equal(observed[steps], start.state[0])
    for step in range(horizon):
        assert seen["reference_action"][step][0] is seen["critic_obs"][step][0]
        assert seen["actor_obs"][step][0] is seen["critic_obs"][step + 1][0]
        assert seen["reward"][step][0] is imagined[step][0]
        torch.testing.assert_close(seen["critic_obs"][step + 1][0], imagined[step][1])


def test_default_transition_hooks_keep_the_given_states() -> None:
    """Without overrides, transitions start from the observed state and keep the integrated successor."""
    task, state, successor = DifferentiableTask(), (torch.randn(4, 1),), (torch.randn(4, 1),)
    assert task.pre_dynamics(state, torch.zeros(4, 1)) is state
    assert task.post_dynamics(state, successor, None, torch.zeros(4, 1), torch.zeros(4, 1)) is successor


class HookedScalarTask(ImpulseScalarTask, HistoryCriticScalarTask, LoadScalarTask):
    """The impulse, critic history and label hooks at once."""
