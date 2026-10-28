# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import Symmetry, resolve_symmetry_config
from rsl_rl.models import MLPModel, QMLPModel
from rsl_rl.modules import EmpiricalNormalization, SquashedGaussianDistribution
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import compile_model, resolve_class, resolve_obs_groups, resolve_optimizer


@dataclass
class _Batch:
    observations: TensorDict
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: TensorDict
    bootstrap_discounts: torch.Tensor


class SAC:
    r"""Soft Actor-Critic with an ensemble of state-action critics and chronological n-step replay.

    Models and storage are explicit dependencies. The default configuration implements scalar SAC; task presets may
    select optional FlashSAC components. Replay is populated under inference mode, while normalizers are updated
    only by :meth:`update`.

    References:
        Haarnoja et al. "Soft Actor-Critic Algorithms and Applications." arXiv:1812.05905 (2018).
        Kim et al. "FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control."
        arXiv:2604.04539 (2026).
    """

    def __init__(
        self,
        actor: MLPModel,
        critics: list[QMLPModel],
        storage: ReplayBuffer,
        num_updates: int = 1,
        batch_size: int = 256,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_update_interval: int = 1,
        learning_starts: int = 1000,
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.0003,
        alpha_learning_rate: float = 0.0003,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        max_grad_norm: float | None = None,
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        target_entropy_std: float | None = None,
        critic_loss: str = "mse",
        num_bins: int = 101,
        value_range: tuple[float, float] = (-5.0, 5.0),
        reward_normalization: bool = False,
        reward_normalization_max_return: float = 5.0,
        exploration: str = "gaussian",
        noise_repeat_exponent: float = 2.0,
        noise_repeat_max: int = 16,
        learning_rate_schedule: str = "constant",
        final_learning_rate_ratio: float = 0.5,
        schedule_updates: int | None = None,
        use_mixed_precision: bool = False,
        symmetry_cfg: dict | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize explicit models, replay, optimizer settings and SAC hyperparameters."""
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Multi-GPU training is not supported for SAC.")
        if not isinstance(actor.distribution, SquashedGaussianDistribution):
            raise ValueError("SAC requires a SquashedGaussianDistribution actor.")
        if actor.is_recurrent or any(critic.is_recurrent for critic in critics):
            raise ValueError("SAC does not support recurrent models.")
        if len(critics) < 2:
            raise ValueError("SAC requires at least two critics.")
        if min(num_updates, batch_size, n_steps, actor_update_interval) < 1 or learning_starts < 0:
            raise ValueError("Update counts and batch/window sizes must be positive; learning_starts nonnegative.")
        if not 0 <= gamma <= 1 or not 0 < tau <= 1 or init_alpha <= 0:
            raise ValueError("SAC requires gamma in [0,1], tau in (0,1] and positive init_alpha.")
        if critic_loss not in ("mse", "categorical") or exploration not in ("gaussian", "held"):
            raise ValueError("Unsupported critic loss or exploration mode.")
        if num_bins < 2 or value_range[0] >= value_range[1]:
            raise ValueError("Categorical critics need at least two bins and an ordered value range.")
        if learning_rate_schedule not in ("constant", "cosine"):
            raise ValueError("Unsupported learning rate schedule.")
        if learning_rate_schedule == "cosine" and (schedule_updates is None or schedule_updates < 1):
            raise ValueError("A cosine schedule requires positive schedule_updates.")
        if reward_normalization_max_return <= 0 or noise_repeat_max < 1:
            raise ValueError("Reward normalization floor and maximum noise repetitions must be positive.")
        if target_entropy_std is not None and target_entropy_std <= 0:
            raise ValueError("target_entropy_std must be positive.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg is not None else None
        self.device = device
        self.actor = actor.to(device)
        self.critics = nn.ModuleList(critics).to(device)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self._raw_actor = self.actor
        self._raw_critics = list(self.critics)
        self.log_alpha = torch.tensor(math.log(init_alpha), device=device, requires_grad=True)
        optimizer_class = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )
        self.critic_optimizer = optimizer_class(
            self.critics.parameters(), lr=critic_learning_rate, weight_decay=weight_decay
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.learning_rate = critic_learning_rate
        self.storage = storage
        self.num_updates, self.batch_size, self.n_steps = num_updates, batch_size, n_steps
        self._discounts = gamma ** torch.arange(n_steps + 1, dtype=torch.float, device=device)
        self.gamma, self.tau = gamma, tau
        self.actor_update_interval, self.learning_starts = actor_update_interval, learning_starts
        self.max_grad_norm, self.use_mixed_precision = max_grad_norm, use_mixed_precision
        self.target_entropy = -float(actor.distribution.output_dim) if target_entropy is None else target_entropy
        if target_entropy is None and target_entropy_std is not None:
            self.target_entropy = (
                0.5 * actor.distribution.output_dim * math.log(2 * math.pi * math.e * target_entropy_std**2)
            )
        self.critic_loss, self.num_bins, self.value_range = critic_loss, num_bins, value_range
        self._bin_values = torch.linspace(*value_range, num_bins, device=device)
        self.reward_normalization = reward_normalization
        self.reward_normalization_max_return = reward_normalization_max_return
        self._return_normalizer = EmpiricalNormalization(1, eps=0.0).to(device)
        self._returns = torch.zeros(storage.num_envs, device=device)
        self._max_abs_return = torch.zeros((), device=device)
        self._pending_returns: list[torch.Tensor] = []
        self.exploration = exploration
        repeats = torch.arange(1, noise_repeat_max + 1, dtype=torch.float, device=device)
        self._noise_repeat_cdf = torch.cumsum(repeats**-noise_repeat_exponent, dim=0)
        self._noise_repeat_cdf /= self._noise_repeat_cdf[-1].clone()
        self._noise = torch.zeros(storage.num_envs, actor.distribution.output_dim, device=device)
        self._noise_steps_left = torch.zeros(storage.num_envs, dtype=torch.long, device=device)
        self._actor_batch_norms = _batch_norm_layers(self.actor)
        self._critic_batch_norms = _batch_norm_layers(self.critics)
        self._target_batch_norms = _batch_norm_layers(self.target_critics)
        self._schedulers = []
        if learning_rate_schedule == "cosine":
            for opt in (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer):
                self._schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt, T_max=schedule_updates, eta_min=opt.param_groups[0]["lr"] * final_learning_rate_ratio
                    )
                )
        self.num_env_steps = 0
        self.num_critic_updates = 0
        self._actor_loaded = False
        self._observations: TensorDict | None = None
        self._actions: torch.Tensor | None = None
        self._pending_normalization: list[TensorDict] = []

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Choose exploratory actions and retain the transition's starting observation."""
        distribution = self.actor.distribution
        # Keep policy statistics available to the runner during uniform-action warm-up.
        with _batch_norm_mode(self._actor_batch_norms, training=False):
            actions = self.actor(obs, stochastic_output=True)
        if not self._actor_loaded and self.num_env_steps * self.storage.num_envs < self.learning_starts:
            uniform = torch.rand(obs.batch_size[0], distribution.output_dim, device=self.device) * 2.0 - 1.0
            actions = distribution.range_offset + distribution.range_scale * uniform
        else:
            if self.exploration == "held":
                resample = self._noise_steps_left <= 0
                self._noise.copy_(torch.where(resample[:, None], torch.randn_like(self._noise), self._noise))
                repeats = torch.searchsorted(self._noise_repeat_cdf, torch.rand_like(self._noise_steps_left.float()))
                self._noise_steps_left.copy_(torch.where(resample, repeats + 1, self._noise_steps_left) - 1)
                actions = distribution.sample(noise=self._noise)
        self._observations, self._actions = obs, actions
        return actions

    def _make_transition(
        self, obs: TensorDict, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> TensorDict:
        """Build replay data with explicit transition-end observations."""
        dones = dones.reshape(-1).bool()
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).reshape(-1).bool()
        groups = list(dict.fromkeys(self.actor.obs_groups + [g for c in self.critics for g in c.obs_groups]))
        next_obs = obs.select(*groups).clone()
        if dones.any():
            if extras.get("final_obs") is None:
                raise ValueError("SAC requires extras['final_obs'] on every step containing a done.")
            final_obs = extras["final_obs"]
            for key in groups:
                mask = dones.reshape(-1, *([1] * (obs[key].ndim - 1)))
                next_obs[key] = torch.where(mask, final_obs[key].to(self.device), obs[key])
        return TensorDict(
            {
                "observations": self._observations.select(*groups),
                "actions": actions,
                "rewards": rewards.reshape(-1),
                "next_observations": next_obs,
                "dones": dones,
                "terminated": dones & ~time_outs,
            },
            batch_size=[self.storage.num_envs],
            device=self.device,
        )

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Store physical successors, replacing all reset rows with their transition-end observations."""
        if self._observations is None or self._actions is None:
            raise RuntimeError("act() must precede process_env_step().")
        self.storage.add(self._make_transition(obs, self._actions, rewards, dones, extras))
        self._pending_normalization.append(self._observations.detach().clone())
        if self.reward_normalization:
            self._returns.mul_(self.gamma).add_(rewards.reshape(-1))
            self._max_abs_return.copy_(torch.maximum(self._max_abs_return, self._returns.abs().max()))
            self._pending_returns.append(self._returns.clone())
            self._returns.mul_(~dones.reshape(-1).bool())
        self._noise_steps_left.mul_(~dones.reshape(-1).bool())
        self.num_env_steps += 1
        self.actor.reset(dones)
        for critic in self.critics:
            critic.reset(dones)
        self._observations = self._actions = None

    def compute_returns(self, obs: TensorDict) -> None:
        """Replay algorithms construct returns from sampled chronological windows during update."""
        pass

    def _sample_batch(self) -> _Batch:
        """Reduce a replay window, bootstrapping partial windows from their last explicit successor."""
        batch = self.storage.sample(self.batch_size, length=self.n_steps)
        rewards = (batch.window["rewards"] * self._discounts[: self.n_steps] * batch.valid).sum(1)
        discounts = self._discounts[batch.valid.sum(1)] * ~batch.terminated
        return _Batch(
            batch.observations,
            batch.actions,
            rewards,
            batch.next_observations,
            discounts,
        )

    def update(self) -> dict[str, float]:
        """Update normalization from newly collected observations, then fit critics, actor and temperature."""
        if self._pending_normalization:
            observations = torch.cat(self._pending_normalization)
            self.actor.update_normalization(observations)
            for critic in self.critics:
                critic.update_normalization(observations)
            self._pending_normalization.clear()
        if self._pending_returns:
            self._return_normalizer.update(torch.cat(self._pending_returns).unsqueeze(-1))
            self._pending_returns.clear()
        if self.storage.size < self.learning_starts or not self.storage.size:
            return {}
        totals = {}
        actor_updates = 0
        for _ in range(self.num_updates):
            batch = self._augment_batch(self._sample_batch())
            if self.reward_normalization:
                batch.rewards = batch.rewards / self._reward_scale()
            do_actor = self.num_critic_updates % self.actor_update_interval == 0
            stats = self._update_actor_and_alpha(batch) if do_actor else {}
            stats.update(self._update_critics(batch))
            self._update_target_critics()
            self.num_critic_updates += 1
            for scheduler in self._schedulers:
                if scheduler.last_epoch < scheduler.T_max:
                    scheduler.step()
            self.learning_rate = float(self.critic_optimizer.param_groups[0]["lr"])
            actor_updates += int(do_actor)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
        return {
            key: (
                value / (actor_updates if key in ("actor", "alpha", "entropy", "symmetry") else self.num_updates)
            ).item()
            for key, value in totals.items()
        }

    def _augment_batch(self, batch: _Batch) -> _Batch:
        """Augment both endpoints with the existing Symmetry extension before constructing replay losses."""
        if self.symmetry is None or not self.symmetry.use_data_augmentation:
            return batch
        augment = self.symmetry.data_augmentation_func
        observations, actions = augment(env=self.symmetry.env, obs=batch.observations, actions=batch.actions)
        successors, _ = augment(env=self.symmetry.env, obs=batch.next_observations, actions=None)
        num_aug = actions.shape[0] // self.batch_size
        return _Batch(
            observations,
            actions,
            batch.rewards.repeat(num_aug),
            successors,
            batch.bootstrap_discounts.repeat(num_aug),
        )

    def _update_critics(self, batch: _Batch) -> dict[str, torch.Tensor]:
        """Run one gradient step of the critics."""
        alpha = self.log_alpha.detach().exp()
        actions = self._critic_actions(batch.actions)
        joint_batch = len(self._critic_batch_norms) > 0

        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
        ):
            with torch.no_grad():
                with _batch_norm_mode(self._actor_batch_norms, training=False):
                    next_actions, next_log_probs = self._sample_actions(batch.next_observations)
                next_actions = self._critic_actions(next_actions)
                if joint_batch:
                    target_obs = torch.cat([batch.observations, batch.next_observations])
                    target_actions = torch.cat([actions, next_actions])
                    with _batch_norm_mode(self._target_batch_norms, training=True):
                        next_outputs = [
                            c(target_obs, target_actions)[batch.actions.shape[0] :] for c in self.target_critics
                        ]
                else:
                    next_outputs = [c(batch.next_observations, next_actions) for c in self.target_critics]
                targets = self._compute_targets(next_outputs, batch, alpha * next_log_probs)

            if joint_batch:
                with _batch_norm_mode(self._critic_batch_norms, training=True):
                    outputs = [c(target_obs, target_actions)[: batch.actions.shape[0]] for c in self.critics]
            else:
                outputs = [c(batch.observations, actions) for c in self.critics]

            if self.critic_loss == "mse":
                losses = [(output.squeeze(-1) - target).square().mean() for output, target in zip(outputs, targets)]
            else:
                losses = [
                    -(target * torch.log_softmax(output.float(), dim=-1)).sum(dim=-1).mean()
                    for output, target in zip(outputs, targets)
                ]
            critic_loss = torch.stack(losses).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        _normalize_parameters(self.critics)

        with torch.no_grad():
            q_values = torch.stack([self._expected_value(output) for output in outputs])
            stats = {"critic": critic_loss.detach(), "q_value": q_values.mean()}
        return stats

    def _update_actor_and_alpha(self, batch: _Batch) -> dict[str, torch.Tensor]:
        """Run one gradient step of the actor and the entropy temperature."""
        alpha = self.log_alpha.detach().exp()
        joint_batch = len(self._actor_batch_norms) > 0

        with torch.autocast(
            device_type=torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision
        ):
            if joint_batch:
                with _batch_norm_mode(self._actor_batch_norms, training=True):
                    actions, log_probs = self._sample_actions(torch.cat([batch.observations, batch.next_observations]))
                actions, log_probs = actions[: batch.actions.shape[0]], log_probs[: batch.actions.shape[0]]
            else:
                actions, log_probs = self._sample_actions(batch.observations)

            self.critics.requires_grad_(False)
            try:
                with _batch_norm_mode(self._critic_batch_norms, training=False):
                    q_values = torch.stack([
                        self._expected_value(c(batch.observations, self._critic_actions(actions))) for c in self.critics
                    ])
            finally:
                self.critics.requires_grad_(True)
            q_value = q_values.min(dim=0).values
            actor_loss = (alpha * log_probs - q_value).mean()
            symmetry_loss = None
            if self.symmetry is not None:
                symmetry_loss = self.symmetry.compute_loss(self.actor, copy.copy(batch), self.batch_size)
                if self.symmetry.use_mirror_loss:
                    actor_loss = actor_loss + self.symmetry.mirror_loss_coeff * symmetry_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        _normalize_parameters(self.actor)

        entropy = -log_probs.detach().float().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        stats = {"actor": actor_loss.detach(), "alpha": alpha_loss.detach(), "entropy": entropy}
        if symmetry_loss is not None:
            stats["symmetry"] = symmetry_loss.detach()
        return stats

    def _compute_targets(
        self, next_outputs: list[torch.Tensor], batch: _Batch, entropy_bonus: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute the critic targets (values for scalar critics, probabilities for categorical critics)."""
        rewards, discounts = batch.rewards, batch.bootstrap_discounts
        next_values = [self._expected_value(output) for output in next_outputs]
        min_ids = torch.stack(next_values).argmin(dim=0)
        selected = torch.stack(next_outputs).gather(0, min_ids.view(1, -1, 1).expand(1, -1, next_outputs[0].shape[-1]))
        next_outputs = [selected[0]] * len(self.critics)
        if self.critic_loss == "mse":
            return [rewards + discounts * (output.squeeze(-1) - entropy_bonus) for output in next_outputs]
        return [
            self._project(torch.softmax(output.float(), dim=-1), rewards, discounts, entropy_bonus)
            for output in next_outputs
        ]

    def _project(
        self, probs: torch.Tensor, rewards: torch.Tensor, discounts: torch.Tensor, entropy_bonus: torch.Tensor
    ) -> torch.Tensor:
        """Project the shifted and scaled categorical distribution onto the fixed support."""
        v_min, v_max = self.value_range
        delta = (v_max - v_min) / (self.num_bins - 1)
        shifted = rewards.unsqueeze(1) + discounts.unsqueeze(1) * (self._bin_values - entropy_bonus.unsqueeze(1))
        positions = (shifted.clamp(v_min, v_max) - v_min) / delta
        lower = positions.floor().long().clamp(0, self.num_bins - 1)
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = positions - lower.float()
        target = torch.zeros_like(probs)
        target.scatter_add_(1, lower, probs * (1.0 - upper_weight))
        target.scatter_add_(1, upper, probs * upper_weight)
        return target

    def _expected_value(self, output: torch.Tensor) -> torch.Tensor:
        """Return the Q-value of a critic output."""
        if self.critic_loss == "mse":
            return output.squeeze(-1).float()
        return (torch.softmax(output.float(), dim=-1) * self._bin_values).sum(dim=-1)

    def _critic_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Express fixed-range actions in normalized coordinates for critic inputs."""
        distribution = self.actor.distribution
        return (actions - distribution.range_offset) / distribution.range_scale

    def _reward_scale(self) -> torch.Tensor:
        """Compute a private discounted-return scale with a maximum-return floor."""
        return torch.maximum(
            self._return_normalizer._std.squeeze() + 1e-8, self._max_abs_return / self.reward_normalization_max_return
        )

    @torch.no_grad()
    def _update_target_critics(self) -> None:
        """Polyak-average parameters and observation normalizers, retaining target BatchNorm statistics."""
        torch._foreach_lerp_(list(self.target_critics.parameters()), list(self.critics.parameters()), self.tau)
        for target, source in zip(self.target_critics, self.critics):
            for target_buffer, source_buffer in zip(target.obs_normalizer.buffers(), source.obs_normalizer.buffers()):
                target_buffer.copy_(source_buffer)

    def _sample_actions(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actor actions and corrected log probabilities."""
        actions = self.actor(obs, stochastic_output=True)
        return actions, self.actor.get_output_log_prob(actions)

    def train_mode(self) -> None:
        """Enable training for actor and critics."""
        self.actor.train()
        self.critics.train()
        self.target_critics.eval()

    def eval_mode(self) -> None:
        """Disable training behavior for inference."""
        self.actor.eval()
        self.critics.eval()
        self.target_critics.eval()

    def get_policy(self) -> MLPModel:
        """Return the actor in its uncompiled form."""
        return self._raw_actor

    def default_load_cfg(self, loaded_dict: dict) -> dict:
        """Return the default full training checkpoint selection."""
        return {"actor": True, "critic": True, "optimizer": True, "iteration": True}

    def save(self) -> dict:
        """Save model, temperature, optimizer and update counters; replay and simulator state are excluded."""
        saved = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": nn.ModuleList(self._raw_critics).state_dict(),
            "target_critic_state_dict": self.target_critics.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "num_env_steps": self.num_env_steps,
            "num_critic_updates": self.num_critic_updates,
            "scheduler_state_dicts": [scheduler.state_dict() for scheduler in self._schedulers],
        }
        if self.reward_normalization:
            saved["return_normalizer_state_dict"] = self._return_normalizer.state_dict()
            saved["max_abs_return"] = self._max_abs_return.clone()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore selected models and states; any loaded actor disables uniform-action warm-up."""
        load_cfg = self.default_load_cfg(loaded_dict) if load_cfg is None else load_cfg
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
            self._actor_loaded = True
        if load_cfg.get("critic"):
            nn.ModuleList(self._raw_critics).load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self.target_critics.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
            if self.reward_normalization:
                self._return_normalizer.load_state_dict(loaded_dict["return_normalizer_state_dict"], strict=strict)
                self._max_abs_return.copy_(loaded_dict["max_abs_return"])
            with torch.no_grad():
                self.log_alpha.copy_(loaded_dict["log_alpha"])
        if load_cfg.get("optimizer"):
            for name in ("actor", "critic", "alpha"):
                getattr(self, f"{name}_optimizer").load_state_dict(loaded_dict[f"{name}_optimizer_state_dict"])
            self.learning_rate = float(self.critic_optimizer.param_groups[0]["lr"])
            for scheduler, state in zip(self._schedulers, loaded_dict.get("scheduler_state_dicts", [])):
                scheduler.load_state_dict(state)
        if load_cfg.get("iteration"):
            self.num_env_steps, self.num_critic_updates = (
                loaded_dict["num_env_steps"],
                loaded_dict["num_critic_updates"],
            )
        return load_cfg.get("iteration", False)

    def compile(self, mode: str | None = None) -> None:
        """Compile model forward passes using the same compile-mode validation as PPO."""
        self.actor = compile_model(self._raw_actor, mode)
        self.critics = nn.ModuleList([compile_model(critic, mode) for critic in self._raw_critics])

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        """Construct explicit configured models and inject lazy replay storage."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        alg_cfg = resolve_symmetry_config(alg_cfg, env)
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        num_critics = alg_cfg.pop("num_critics", 2)
        critic_dim = alg_cfg.get("num_bins", 101) if alg_cfg.get("critic_loss", "mse") == "categorical" else 1
        critics = [
            critic_class(obs, cfg["obs_groups"], "critic", critic_dim, env.num_actions, **critic_cfg).to(device)
            for _ in range(num_critics)
        ]
        storage = ReplayBuffer(
            env.num_envs,
            alg_cfg.pop("replay_buffer_size", 100_000),
            device,
            min_steps=max(alg_cfg.get("n_steps", 1), math.ceil(alg_cfg.get("learning_starts", 1000) / env.num_envs)),
            max_memory_fraction=alg_cfg.pop("max_replay_memory_fraction", None),
        )
        alg = alg_class(actor, critics, storage, device=device, multi_gpu_cfg=cfg["multi_gpu"], **alg_cfg)
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


def _batch_norm_layers(module: nn.Module) -> list[nn.Module]:
    """Return the batch normalization layers of a module."""
    return [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


class _batch_norm_mode:  # noqa: N801
    """Context manager that temporarily sets the mode of batch normalization layers."""

    def __init__(self, layers: list[nn.Module], training: bool) -> None:
        self.layers = layers
        self.training = training

    def __enter__(self) -> None:
        self.previous = [layer.training for layer in self.layers]
        for layer in self.layers:
            layer.training = self.training

    def __exit__(self, *args: object) -> None:
        for layer, training in zip(self.layers, self.previous):
            layer.training = training


def _normalize_parameters(module: nn.Module) -> None:
    """Apply weight normalization of networks that define it after an optimizer step."""
    for m in module.modules():
        if m is not module and hasattr(m, "normalize_parameters"):
            m.normalize_parameters()
