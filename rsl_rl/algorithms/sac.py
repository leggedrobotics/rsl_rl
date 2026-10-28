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
from rsl_rl.models import MLPModel, QMLPModel
from rsl_rl.modules import SquashedGaussianDistribution
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
    """Soft Actor-Critic with scalar state-action critics and chronological n-step replay."""

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
        use_mixed_precision: bool = False,
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
        actions = self.actor(obs, stochastic_output=True)
        if not self._actor_loaded and self.num_env_steps * self.storage.num_envs < self.learning_starts:
            uniform = torch.rand(obs.batch_size[0], distribution.output_dim, device=self.device) * 2.0 - 1.0
            actions = distribution.range_offset + distribution.range_scale * uniform
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
        if self.storage.size < self.learning_starts or not self.storage.size:
            return {}
        totals = {}
        actor_updates = 0
        for _ in range(self.num_updates):
            batch = self._sample_batch()
            do_actor = self.num_critic_updates % self.actor_update_interval == 0
            stats = self._update_actor_and_alpha(batch) if do_actor else {}
            stats.update(self._update_critics(batch))
            self._update_target_critics()
            self.num_critic_updates += 1
            actor_updates += int(do_actor)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
        return {
            key: (
                value / (actor_updates if key in ("actor", "alpha", "entropy", "symmetry") else self.num_updates)
            ).item()
            for key, value in totals.items()
        }

    def _update_critics(self, batch: _Batch) -> dict[str, torch.Tensor]:
        """Fit scalar critics against entropy-regularized target values."""
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            with torch.no_grad():
                next_actions, log_probs = self._sample_actions(batch.next_observations)
                next_values = (
                    torch
                    .stack([
                        critic(batch.next_observations, self._critic_actions(next_actions)).squeeze(-1)
                        for critic in self.target_critics
                    ])
                    .min(dim=0)
                    .values
                )
                targets = batch.rewards + batch.bootstrap_discounts * (next_values - self.log_alpha.exp() * log_probs)
            outputs = [
                critic(batch.observations, self._critic_actions(batch.actions)).squeeze(-1) for critic in self.critics
            ]
            loss = torch.stack([(output - targets).square().mean() for output in outputs]).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        return {"critic": loss.detach(), "q_value": torch.stack(outputs).detach().mean()}

    def _update_actor_and_alpha(self, batch: _Batch) -> dict[str, torch.Tensor]:
        """Update the reparameterized actor and learned entropy temperature."""
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            actions, log_probs = self._sample_actions(batch.observations)
            self.critics.requires_grad_(False)
            try:
                q_values = (
                    torch
                    .stack([
                        critic(batch.observations, self._critic_actions(actions)).squeeze(-1) for critic in self.critics
                    ])
                    .min(dim=0)
                    .values
                )
            finally:
                self.critics.requires_grad_(True)
            loss = (self.log_alpha.detach().exp() * log_probs - q_values).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        entropy = -log_probs.detach().float().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return {"actor": loss.detach(), "alpha": alpha_loss.detach(), "entropy": entropy}

    def _critic_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Express fixed-range actions in normalized coordinates for critic inputs."""
        distribution = self.actor.distribution
        return (actions - distribution.range_offset) / distribution.range_scale

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
        }
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
            with torch.no_grad():
                self.log_alpha.copy_(loaded_dict["log_alpha"])
        if load_cfg.get("optimizer"):
            for name in ("actor", "critic", "alpha"):
                getattr(self, f"{name}_optimizer").load_state_dict(loaded_dict[f"{name}_optimizer_state_dict"])
            self.learning_rate = float(self.critic_optimizer.param_groups[0]["lr"])
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
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        num_critics = alg_cfg.pop("num_critics", 2)
        critic_dim = 1
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
