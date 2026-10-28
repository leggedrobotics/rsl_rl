# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic world-model fitting and differentiable policy optimization."""

from __future__ import annotations

import math
import torch
from collections.abc import Generator
from contextlib import contextmanager
from tensordict import TensorDict
from torch import nn

from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.models.supermimic_actor import SuperMimicActor
from rsl_rl.modules.world_model import WorldModel
from rsl_rl.storage import SequenceReplayBuffer
from rsl_rl.utils import compile_model, resolve_class, resolve_obs_groups


def lambda_returns(
    rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute TD(lambda) targets, retaining terminal rewards and the final bootstrap state."""
    target, targets = values[-1], []
    for index in reversed(range(rewards.shape[0])):
        target = rewards[index] + gamma * (1 - dones[index]) * ((1 - lam) * values[index + 1] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1])


class SuperMimic:
    """Fit physical dynamics and optimize a goal-conditioned actor through imagined trajectories.

    The environment supplies noise-free state, goal, context and event-label observation groups. Every-step final
    observations preserve the physical successor before reset and interval events. The task owns geometry, rewards,
    analytic state updates and deployment features.

    Reference:
        SuperMimic. arXiv:2610.28480 (2026). https://arxiv.org/abs/2610.28480
    """

    def __init__(
        self,
        actor: SuperMimicActor,
        world_model: WorldModel,
        critic: MLPModel,
        storage: SequenceReplayBuffer,
        task: DifferentiableTask,
        obs_groups: dict[str, list[str]],
        updates_per_step: float = 0.4,
        num_world_model_updates: int = 10,
        num_policy_updates: int = 1,
        world_model_window: int = 8,
        world_model_batch_size: int = 2048,
        horizon: int = 32,
        policy_batch_size: int = 4096,
        learning_rate: float = 2e-4,
        lr_decay_iterations: int = 250_000,
        lr_final_ratio: float = 0.03,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
        warmup_steps: int = 64,
        world_model_warmup_iterations: int = 500,
        action_noise: float = 0.1,
        max_step_reward: float | None = None,
        use_mixed_precision: bool = False,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize explicit task, models, replay and gradient-update cadence.

        Actor gradients also have a fixed elementwise clip of 0.1. The actor Adam optimizer uses betas (0.9, 0.95).

        Args:
            actor: History-conditioned deployment actor.
            world_model: Differentiable physical increment and optional event predictor.
            critic: Scalar value model consuming synthetic task features.
            storage: Lazy replay holding complete physical transitions.
            task: Differentiable task heads and physical state coordinates.
            obs_groups: Explicit physical observation-set mapping.
            updates_per_step: Learner iterations accrued per collected environment step.
            num_world_model_updates: Dynamics optimizer steps per learner iteration.
            num_policy_updates: Actor updates and associated critic fits per learner iteration.
            world_model_window: Maximum autoregressive dynamics fitting horizon.
            world_model_batch_size: Replay anchors per dynamics optimizer step.
            horizon: Imagined policy horizon.
            policy_batch_size: Starts per actor update.
            learning_rate: Initial learning rate for all optimizers.
            lr_decay_iterations: Learner iterations in the shared cosine learning-rate schedule.
            lr_final_ratio: Final learning rate relative to its initial value.
            gamma: Reward discount.
            gae_lambda: TD(lambda) mixing coefficient for value targets.
            critic_epochs: Epochs per critic fit.
            critic_minibatches: Minibatches per critic epoch.
            warmup_steps: Stored steps per environment required before learning.
            world_model_warmup_iterations: Learner iterations fitting only dynamics.
            action_noise: Standard deviation of additive collection noise.
            max_step_reward: Optional per-step reward bound used to cap values at the maximum discounted return.
            use_mixed_precision: Enable BF16 matrix operations while integrating physical states in FP32.
            max_grad_norm: Global gradient-norm limit for each model.
            device: Device for models and learning tensors.
            multi_gpu_cfg: Unsupported distributed configuration; supplying one raises.
        """
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Multi-GPU training is not supported for SuperMimic.")
        if critic.is_recurrent:
            raise ValueError("SuperMimic requires a feed-forward critic.")
        positive = (
            num_world_model_updates,
            num_policy_updates,
            world_model_window,
            world_model_batch_size,
            horizon,
            policy_batch_size,
            lr_decay_iterations,
            critic_epochs,
            critic_minibatches,
            warmup_steps,
        )
        if min(positive) < 1 or updates_per_step <= 0 or learning_rate <= 0 or max_grad_norm <= 0:
            raise ValueError("Learning counts, cadence, learning rate and gradient limit must be positive.")
        if warmup_steps < world_model_window or world_model_warmup_iterations < 0:
            raise ValueError("warmup_steps must cover the model window; world-model warmup must be nonnegative.")
        if not 0 <= gamma < 1 or not 0 <= gae_lambda <= 1 or not 0 <= lr_final_ratio <= 1:
            raise ValueError("Invalid discount, lambda or final learning-rate ratio.")
        if action_noise < 0:
            raise ValueError("Action noise must be nonnegative.")
        self.actor, self.world_model, self.critic = actor.to(device), world_model.to(device), critic.to(device)
        self._raw_critic = self.critic
        self.storage, self.task, self.obs_groups = storage, task, obs_groups
        self.device = device
        self.updates_per_step = updates_per_step
        self.num_world_model_updates, self.num_policy_updates = num_world_model_updates, num_policy_updates
        self.world_model_window, self.world_model_batch_size = world_model_window, world_model_batch_size
        self.horizon, self.policy_batch_size = horizon, policy_batch_size
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.critic_epochs, self.critic_minibatches = critic_epochs, critic_minibatches
        self.warmup_steps, self.world_model_warmup_iterations = warmup_steps, world_model_warmup_iterations
        self.value_cap = None if max_step_reward is None else max_step_reward / (1 - gamma)
        self.use_mixed_precision, self.max_grad_norm = use_mixed_precision, max_grad_norm
        self.actor.noise_std.fill_(action_noise)
        self.error_floor = task.representation.error_floor.to(device)
        if self.error_floor.shape != (task.representation.increment_dim,) or (self.error_floor <= 0).any():
            raise ValueError("State error_floor must contain one positive value per increment coordinate.")
        optimizer_kwargs = {"lr": learning_rate, "fused": torch.device(device).type == "cuda"}
        self.optimizers = {
            "world_model": torch.optim.Adam(self.world_model.parameters(), **optimizer_kwargs),
            "actor": torch.optim.Adam(self.actor.parameters(), betas=(0.9, 0.95), **optimizer_kwargs),
            "critic": torch.optim.Adam(self.critic.parameters(), **optimizer_kwargs),
        }
        self.initial_learning_rate = learning_rate
        self.lr_decay_iterations, self.lr_final_ratio = lr_decay_iterations, lr_final_ratio
        self.learning_rate = learning_rate
        self.iteration, self._update_debt = 0, 0.0
        self._last_obs: TensorDict | None = None
        self._last_actions: torch.Tensor | None = None
        self._last_previous_actions: torch.Tensor | None = None
        self._window_break_sum = torch.zeros((), device=device)
        self._collected_steps = 0
        self._world_model_forward = self.world_model
        self._actor_forward = self.actor.forward_history

    def _set(self, obs: TensorDict, name: str) -> torch.Tensor:
        """Read an explicit observation set with no policy-group fallback."""
        return self.actor.observation_set(obs, name)

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Record the physical start and choose a noisy actor action; update actor moments now."""
        self.actor.sync_actions(obs)
        self._last_previous_actions = self.actor.previous_actions.clone()
        keys = {key for name in ("state", "goal", "context") for key in self.obs_groups.get(name, ())}
        self._last_obs = obs.select(*keys).clone()
        action = self.actor(obs)
        self._last_actions = action + torch.randn_like(action) * self.actor.noise_std
        self.actor.previous_actions[:, -1].copy_(self._last_actions)
        return self._last_actions

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record every-step pre-event successors and update dynamics statistics in place."""
        if self._last_obs is None or self._last_actions is None:
            raise RuntimeError("act() must precede process_env_step().")
        if extras.get("final_obs") is None or extras.get("final_obs_every_step") is not True:
            raise ValueError("SuperMimic requires every-step extras['final_obs'] with final_obs_every_step=True.")
        final = extras["final_obs"].to(self.device)
        state, successor = self._set(self._last_obs, "state"), self._set(final, "state")
        context, goal = self._set(self._last_obs, "context"), self._set(self._last_obs, "goal")
        done = dones.reshape(-1).bool()
        window_break = (self._set(obs, "state") != successor).any(dim=-1) & ~done
        transition = TensorDict(
            {
                "state": state,
                "next_state": successor,
                "goal": goal,
                "context": context,
                "actions": self._last_actions,
                "prev_actions": self._last_previous_actions,
                "dones": done,
                "window_break": window_break,
            },
            batch_size=[self.storage.num_envs],
            device=self.device,
        )
        if self.task.num_labels:
            transition["labels"] = self._set(final, "labels")
        self.storage.add(transition)
        start = self.task.representation.unflatten(state)
        self.world_model.observe(
            start,
            self._last_actions,
            self.task.dynamics_context(start, context, self._last_previous_actions),
            self.task.representation.unflatten(successor),
        )
        reset_action = self._set(obs, "prev_action") if "prev_action" in self.obs_groups else None
        reset_previous = self._set(obs, "prev_prev_action") if "prev_prev_action" in self.obs_groups else None
        self.actor.reset(done, reset_action, reset_previous)
        self._window_break_sum.add_(window_break.float().mean())
        self._collected_steps += 1
        self._update_debt += self.updates_per_step
        self._last_obs = self._last_actions = self._last_previous_actions = None

    def compute_returns(self, obs: TensorDict) -> None:
        """Replay and imagination construct their own return targets during update."""
        pass

    def _windows(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Sample uniform anchors, additionally cutting world-model targets after external state edits."""
        index = self.storage.sample(batch_size, length=self.world_model_window)
        keys = ["state", "next_state", "actions", "context", "prev_actions", "window_break"]
        if self.task.num_labels:
            keys.append("labels")
        window = self.storage.gather(keys, index.ids)
        mask = index.valid.float()
        if self.world_model_window > 1:
            mask[:, 1:] *= torch.cumprod((~window["window_break"][:, :-1]).long(), dim=1)
        return window, mask

    def _starts(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Build reset-aware actor histories, repeating the earliest available frame when needed."""
        index = self.storage.sample(batch_size, history=self.actor.history_length - 1)
        history = self.storage.gather(("state", "goal", "context", "prev_actions"), index.ids)
        state = self.task.representation.unflatten(history["state"])
        frames = self.task.add_noise(
            self.task.actor_obs(state, history["goal"], history["context"], history["prev_actions"])
        )
        return history[:, -1], frames

    def _predict(
        self, state: State, actions: torch.Tensor, context: torch.Tensor, previous_actions: torch.Tensor
    ) -> tuple[State, torch.Tensor | None, torch.Tensor | None]:
        """Apply learned dynamics, event probabilities and task-owned analytic state."""
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            raw_next = self._world_model_forward(
                state, actions, self.task.dynamics_context(state, context, previous_actions)
            )
            logits = (
                self.world_model.label_logits(raw_next, self.task.dynamics_context(raw_next, context, previous_actions))
                if self.task.num_labels
                else None
            )
        labels = torch.sigmoid(logits.float()) if logits is not None else None
        return self.task.post_dynamics(state, raw_next, labels), logits, labels

    def _fit_world_model(self) -> dict[str, torch.Tensor]:
        """Fit autoregressive physical successors and event labels on valid replay positions."""
        window, mask = self._windows(self.world_model_batch_size)
        state = self.task.representation.unflatten(window["state"][:, 0])
        predictions, logits = [], []
        for step in range(self.world_model_window):
            state, event_logits, _ = self._predict(
                state, window["actions"][:, step], window["context"][:, step], window["prev_actions"][:, step]
            )
            predictions.append(state)
            if event_logits is not None:
                logits.append(event_logits)
        prediction = tuple(torch.stack(values, dim=1) for values in zip(*predictions))
        target = self.task.representation.unflatten(window["next_state"])
        scale = torch.maximum(self.world_model.out_norm.std, self.error_floor)
        errors = (self.task.representation.error(prediction, target) / scale).square().mean(dim=-1)
        loss = (errors * mask).sum(dim=1).mean()
        metrics = {"world_model_error": (errors.detach() * mask).sum() / mask.sum().clamp(min=1)}
        if self.task.num_labels:
            bce = nn.functional.binary_cross_entropy_with_logits(
                torch.stack(logits, 1).float(), window["labels"].float(), reduction="none"
            )
            label_loss = (bce.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + label_loss
            metrics["label_loss"] = label_loss.detach()
        self._apply(loss, "world_model", self.world_model)
        metrics["world_model_loss"] = loss.detach() / self.world_model_window
        return metrics

    def _value(self, features: torch.Tensor) -> torch.Tensor:
        """Read scalar values through the standard synthetic critic observation set."""
        return self.critic(TensorDict({"critic": features}, batch_size=features.shape[:-1])).squeeze(-1)

    def _cap(self, values: torch.Tensor) -> torch.Tensor:
        """Clamp values above ``value_cap`` when set; there is no lower bound."""
        return values if self.value_cap is None else values.clamp(max=self.value_cap)

    def _fit_policy(self) -> dict[str, torch.Tensor]:
        """Backpropagate imagined returns through frozen dynamics and fit a TD(lambda) critic."""
        with torch.no_grad():
            start, history = self._starts(self.policy_batch_size)
        state = self.task.representation.unflatten(start["state"])
        goal, context, previous = start["goal"], start["context"], start["prev_actions"]
        rewards, terminals, features = [], [], []
        with _frozen(self.world_model, self.critic):
            for _ in range(self.horizon):
                features.append(self.task.critic_obs(state, goal, context, previous))
                with torch.autocast(
                    torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
                ):
                    action = self._actor_forward(history, self.task.actor_goal(state, goal))
                successor, _, labels = self._predict(state, action, context, previous)
                rewards.append(self.task.reward(state, action, successor, goal, previous, labels=labels))
                terminals.append(self.task.termination_prob(successor, goal, labels=labels))
                goal, context = self.task.advance(successor, goal, context)
                previous = torch.cat((previous[:, 1:], action[:, None]), dim=1)
                frame = self.task.add_noise(self.task.actor_obs(successor, goal, context, previous))
                history = torch.cat((history[:, 1:], frame[:, None]), dim=1)
                state = successor
            features.append(self.task.critic_obs(state, goal, context, previous))
            reward, terminal = torch.stack(rewards), torch.stack(terminals)
            alive = torch.cumprod(1 - terminal, dim=0)
            weights = (
                torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
                * self.gamma ** torch.arange(self.horizon, device=self.device)[:, None]
            )
            value = self._value(features[-1])
            objective = (weights * reward).sum(dim=0) + self.gamma**self.horizon * alive[-1] * self._cap(value)
            policy_loss = -objective.mean()
            gradient = self._apply(policy_loss, "actor", self.actor, clip_value=0.1)
        features_tensor = torch.stack(features).detach()
        metrics = {
            "policy_loss": policy_loss.detach(),
            "policy_grad_norm": gradient,
            "imagined_survival": alive[-1].mean().detach(),
            "imagined_reward": reward.detach().mean(),
        }
        if self.value_cap is not None:
            metrics["terminal_saturation"] = (value.detach() > self.value_cap).float().mean()
        metrics.update(self._fit_critic(features_tensor, reward.detach(), terminal.detach()))
        return metrics

    def _fit_critic(
        self, features: torch.Tensor, rewards: torch.Tensor, terminals: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Fit scalar values on detached imagined features with TD(lambda) targets."""
        with torch.no_grad():
            self.critic.update_normalization(
                TensorDict({"critic": features.flatten(0, 1)}, batch_size=[features.shape[0] * features.shape[1]])
            )
            targets = lambda_returns(rewards, terminals, self._cap(self._value(features)), self.gamma, self.gae_lambda)
            targets = self._cap(targets).flatten()
        features = features[:-1].flatten(0, 1)
        losses = []
        for _ in range(self.critic_epochs):
            for indices in torch.randperm(features.shape[0], device=self.device).chunk(self.critic_minibatches):
                with torch.autocast(
                    torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
                ):
                    loss = (self._value(features[indices]) - targets[indices]).square().mean()
                self._apply(loss, "critic", self.critic)
                losses.append(loss.detach())
        return {"critic_loss": torch.stack(losses).mean()}

    def _apply(self, loss: torch.Tensor, name: str, module: nn.Module, clip_value: float | None = None) -> torch.Tensor:
        """Apply finite gradients with optional value clipping."""
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
        if clip_value is not None:
            nn.utils.clip_grad_value_(module.parameters(), clip_value)
        if torch.isfinite(norm):
            optimizer.step()
        return norm.detach()

    def update(self) -> dict[str, float]:
        """Run accumulated learner updates; the runner's collection cadence remains external."""
        metrics = {"learner_updates": float(self.iteration)}
        if self._collected_steps:
            metrics["window_break_fraction"] = (self._window_break_sum / self._collected_steps).item()
            self._window_break_sum.zero_()
            self._collected_steps = 0
        if self.storage.num_steps < self.warmup_steps:
            self._update_debt = 0.0
            return metrics
        count = int(self._update_debt + 1e-9)
        self._update_debt -= count
        totals: dict[str, list[torch.Tensor]] = {}
        for _ in range(count):
            for _ in range(self.num_world_model_updates):
                for key, value in self._fit_world_model().items():
                    totals.setdefault(key, []).append(value)
            if self.iteration >= self.world_model_warmup_iterations:
                for _ in range(self.num_policy_updates):
                    for key, value in self._fit_policy().items():
                        totals.setdefault(key, []).append(value)
            self.iteration += 1
            self._update_learning_rate()
        metrics.update({key: torch.stack(values).mean().item() for key, values in totals.items()})
        metrics["learner_updates"] = float(self.iteration)
        return metrics

    def _update_learning_rate(self) -> None:
        """Set all optimizer rates for the next learner iteration, clamping at the cosine floor."""
        progress = min(self.iteration / self.lr_decay_iterations, 1.0)
        self.learning_rate = self.initial_learning_rate * (
            self.lr_final_ratio + (1 - self.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = self.learning_rate

    def train_mode(self) -> None:
        """Enable training models; real collection moments update in actor/process calls."""
        self.actor.train()
        self.world_model.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Select deterministic deployment inference without updating actor moments."""
        self.actor.eval()
        self.world_model.eval()
        self.critic.eval()

    def get_policy(self) -> SuperMimicActor:
        """Return the real-observation actor with stateless deployable export methods."""
        return self.actor

    def default_load_cfg(self, loaded_dict: dict) -> dict:
        """Select an explicit full learner checkpoint, including the world model."""
        return {"actor": True, "critic": True, "world_model": True, "optimizer": True, "iteration": True}

    def save(self) -> dict:
        """Save all trainable state; replay and simulator collection state are deliberately excluded."""
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            **{f"{key}_optimizer_state_dict": optimizer.state_dict() for key, optimizer in self.optimizers.items()},
            "iteration": self.iteration,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load an actor alone or a complete model selection; partial dynamics/critic restores are rejected."""
        load_cfg = self.default_load_cfg(loaded_dict) if load_cfg is None else load_cfg
        core = {name for name in ("actor", "critic", "world_model") if load_cfg.get(name)}
        if core not in ({"actor"}, {"actor", "critic", "world_model"}):
            raise ValueError("SuperMimic supports actor-only or full actor/critic/world_model loading.")
        actor_only = core == {"actor"}
        if actor_only and any(load_cfg.get(key) for key in ("optimizer", "iteration")):
            raise ValueError("Actor-only loading cannot restore optimizer or iteration state.")
        if not actor_only and "world_model_state_dict" not in loaded_dict:
            raise ValueError("A full SuperMimic checkpoint must explicitly contain world_model.")
        self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if not actor_only:
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self.world_model.load_state_dict(loaded_dict["world_model_state_dict"], strict=strict)
            if load_cfg.get("optimizer"):
                for key in self.optimizers:
                    self.optimizers[key].load_state_dict(loaded_dict[f"{key}_optimizer_state_dict"])
            if load_cfg.get("iteration"):
                self.iteration = loaded_dict["iteration"]
        self.storage.clear()
        self.actor.reset()
        self._last_obs = self._last_actions = self._last_previous_actions = None
        self._update_debt = 0.0
        self._window_break_sum.zero_()
        self._collected_steps = 0
        self._update_learning_rate()
        return bool(load_cfg.get("iteration"))

    def compile(self, mode: str | None = None) -> None:
        """Compile differentiable dynamics and stateless actor forwards using the standard mode restrictions."""
        self._world_model_forward = compile_model(self.world_model, mode)
        self.critic = compile_model(self._raw_critic, mode)
        self._actor_forward = (
            self.actor.forward_history if mode is None else torch.compile(self.actor.forward_history, mode=mode)
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SuperMimic:
        """Resolve the task and explicit network configs, inferring synthetic feature widths once."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        task_class, task_cfg = resolve_class(alg_cfg.pop("task_cfg"))
        if "critic" in cfg["obs_groups"]:
            raise ValueError("SuperMimic critic observations are synthetic; omit 'critic' from obs_groups.")
        allowed = ("state", "goal", "context", "labels", "prev_action", "prev_prev_action")
        groups = resolve_obs_groups(obs, cfg["obs_groups"], ["state"], fallback_to_policy=False)
        if set(groups) - set(allowed):
            raise ValueError("Unknown SuperMimic observation set.")
        cfg["obs_groups"] = groups
        task = task_class(env=env, device=device, **task_cfg)
        example = obs.to(device)

        def observation_set(name: str) -> torch.Tensor:
            return (
                torch.cat([example[key] for key in groups[name]], dim=-1)
                if name in groups
                else example[groups["state"][0]].new_empty(env.num_envs, 0)
            )

        for name in groups:
            if any(example[key].ndim != 2 for key in groups[name]):
                raise ValueError("SuperMimic observation groups must be flat noise-free vectors.")
        previous = torch.zeros(env.num_envs, task.action_history_length, env.num_actions, device=device)
        for name, offset in (("prev_action", -1), ("prev_prev_action", -2)):
            if name in groups:
                if observation_set(name).shape[-1] != env.num_actions or task.action_history_length < -offset:
                    raise ValueError("Previous-action observation width/history is incompatible.")
                previous[:, offset] = observation_set(name)
        state, goal, context = (
            task.representation.unflatten(observation_set("state")),
            observation_set("goal"),
            observation_set("context"),
        )
        actor_obs, actor_goal = task.actor_obs(state, goal, context, previous), task.actor_goal(state, goal)
        critic_features = task.critic_obs(state, goal, context, previous)
        dynamics_context = task.dynamics_context(state, context, previous)
        if observation_set("labels").shape[-1] != task.num_labels:
            raise ValueError("Label observation width must equal task.num_labels.")
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        model_class, model_cfg = resolve_class(cfg["world_model"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        actor = actor_class(
            task, groups, actor_obs.shape[-1], actor_goal.shape[-1], env.num_envs, env.num_actions, **actor_cfg
        ).to(device)
        world_model = model_class(
            task.representation,
            task.representation.features(state).shape[-1],
            env.num_actions,
            dynamics_context.shape[-1],
            num_labels=task.num_labels,
            **model_cfg,
        ).to(device)
        critic_obs = TensorDict({"critic": critic_features}, batch_size=[env.num_envs])
        critic = critic_class(critic_obs, {"critic": ["critic"]}, "critic", 1, **critic_cfg).to(device)
        warmup = alg_cfg.get("warmup_steps", 64)
        storage = SequenceReplayBuffer(
            env.num_envs,
            alg_cfg.pop("replay_buffer_size", 1_000_000),
            device,
            min_steps=max(warmup, actor.history_length),
            max_memory_fraction=alg_cfg.pop("max_replay_memory_fraction", None),
        )
        learner = alg_class(
            actor,
            world_model,
            critic,
            storage,
            task,
            groups,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **alg_cfg,
        )
        learner.compile(cfg.get("torch_compile_mode"))
        return learner


@contextmanager
def _frozen(*modules: nn.Module) -> Generator[None, None, None]:
    """Freeze parameters while preserving input derivatives and restoring flags after failures."""
    previous = [(parameter, parameter.requires_grad) for module in modules for parameter in module.parameters()]
    try:
        for parameter, _ in previous:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in previous:
            parameter.requires_grad_(enabled)
