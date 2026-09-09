# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code ported from FlashSAC (flash_rl):
#   Copyright (c) 2026 Holiday Robotics
#   SPDX-License-Identifier: MIT
# The FlashSAC update math (categorical distributional TD target, clipped double-Q,
# entropy-temperature tuning, EMA target, noise-repetition exploration) is
# adapted from ``flash_rl/agents/flashSAC/{agent,update}.py``. The RSL-RL
# algorithm contract (act / process_env_step / compute_returns / update /
# save / load / get_policy / construct_algorithm) mirrors ``rsl_rl/algorithms/ppo.py``.

"""Soft Actor-Critic (FlashSAC variant).

An off-policy, entropy-regularized actor-critic that plugs into RSL-RL through
the same algorithm contract as :class:`~rsl_rl.algorithms.PPO`, but is driven by
:class:`~rsl_rl.runners.OffPolicyRunner` instead of the on-policy runner.

Distinctive features (ported from FlashSAC): a squashed-Gaussian actor with
weight-normalized layers, an ensembled categorical (C51-style) double critic
with an EMA target, an auto-tuned entropy temperature, n-step categorical TD
targets, delayed policy updates, and zeta-distributed action-noise repetition
for exploration. This v1 integration is eager-only (no ``torch.compile`` / AMP)
and single-GPU.
"""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import FlashSACActorModel, FlashSACCriticModel
from rsl_rl.modules import FlashSACTemperature
from rsl_rl.storage import ReplayStorage
from rsl_rl.utils import resolve_callable, resolve_sac_obs_groups
from rsl_rl.utils.reward_normalization import RewardNormalizer
from rsl_rl.utils.scheduler import warmup_cosine_decay_scheduler
from rsl_rl.utils.utils import _require


class FlashSAC:
    """Soft Actor-Critic algorithm (FlashSAC variant)."""

    actor: FlashSACActorModel
    """The actor model (policy)."""

    critic: FlashSACCriticModel
    """The distributional double-Q critic."""

    def __init__(
        self,
        actor: FlashSACActorModel,
        critic: FlashSACCriticModel,
        target_critic: FlashSACCriticModel,
        temperature: FlashSACTemperature,
        storage: ReplayStorage,
        *,
        gamma: float,
        n_step: int,
        learning_rate_init: float,
        learning_rate_peak: float,
        learning_rate_end: float,
        learning_rate_warmup_steps: int,
        learning_rate_decay_steps: int,
        critic_target_update_tau: float,
        num_bins: int,
        min_v: float,
        max_v: float,
        temp_target_entropy: float,
        actor_update_period: int,
        actor_bc_alpha: float,
        actor_noise_zeta_mu: float,
        actor_noise_zeta_max: int,
        normalize_reward: bool,
        normalized_g_max: float,
        finite_horizon: bool,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the FlashSAC algorithm with models, storage, and optimization settings."""
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if self.is_multi_gpu:
            raise NotImplementedError(
                "Multi-GPU training is not supported for FlashSAC in this version. Run single-GPU."
            )

        if not normalize_reward:
            raise ValueError(
                "FlashSAC (v1) requires normalize_reward=True: the distributional critic support "
                f"[min_v, max_v]=[{min_v}, {max_v}] assumes reward-normalized returns."
            )

        # Models.
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = target_critic.to(self.device)
        self.temperature = temperature.to(self.device)
        # Handles to the (uncompiled) modules for state_dict/export.
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self._raw_target_critic = self.target_critic

        # Optimizers + warmup-cosine-decay schedules (one per learnable network).
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_peak)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_peak)
        self.temperature_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=learning_rate_peak)
        lr_fn = warmup_cosine_decay_scheduler(
            init_value=learning_rate_init,
            peak_value=learning_rate_peak,
            end_value=learning_rate_end,
            warmup_steps=learning_rate_warmup_steps,
            decay_steps=learning_rate_decay_steps,
        )

        def lr_lambda(step: int) -> float:
            return lr_fn(step) / learning_rate_peak

        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lr_lambda)
        self.critic_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lr_lambda)
        self.temperature_scheduler = torch.optim.lr_scheduler.LambdaLR(self.temperature_optimizer, lr_lambda=lr_lambda)

        # Storage.
        self.storage = storage

        # Hyperparameters.
        self.gamma = gamma
        self.n_step = n_step
        self.critic_target_update_tau = critic_target_update_tau
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v
        self.temp_target_entropy = temp_target_entropy
        self.actor_update_period = actor_update_period
        self.actor_bc_alpha = actor_bc_alpha
        self.normalize_reward = normalize_reward
        self.finite_horizon = finite_horizon

        # Weight normalization is applied after every optimizer step.
        self.actor.normalize_parameters()
        self.critic.normalize_parameters()
        self.target_critic.normalize_parameters()

        # Reward normalizer (required for v1).
        self.reward_normalizer = RewardNormalizer(
            gamma=gamma,
            g_max=normalized_g_max,
            num_envs=storage.num_envs,
            device=torch.device(self.device),
        )

        # Noise-repetition (zeta distribution) exploration state, one per env.
        num_envs = storage.num_envs
        action_dim = storage.actions.shape[-1]
        self._zeta_cdf = self._build_truncated_zeta_cdf(actor_noise_zeta_mu, actor_noise_zeta_max).to(self.device)
        self._cur_noise_n = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self._cur_noise_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._cached_noise = torch.randn(num_envs, action_dim, device=self.device)

        # Transition scratch (populated by act(), consumed by process_env_step()).
        self._transition_obs: TensorDict | None = None
        self._transition_actions: torch.Tensor | None = None

        self._update_step = 0

    # -- exploration helpers -------------------------------------------------

    @staticmethod
    def _build_truncated_zeta_cdf(mu: float, max_n: int) -> torch.Tensor:
        """Build the CDF of a zeta(mu) distribution truncated to ``[1, max_n]``."""
        ns = torch.arange(1, max_n + 1, dtype=torch.float32)
        pmf = ns ** (-mu)
        pmf = pmf / torch.sum(pmf)
        return torch.cumsum(pmf, dim=0)

    def _sample_noise_repeat_n(self, num: int) -> torch.Tensor:
        """Sample ``num`` repeat counts from the truncated zeta CDF."""
        u = torch.rand(num, 1, device=self.device)
        return (u < self._zeta_cdf.unsqueeze(0)).float().argmax(dim=1).long() + 1

    # -- rollout -------------------------------------------------------------

    def act(self, obs: TensorDict, training: bool = True) -> torch.Tensor:
        """Sample exploratory actions and stash the transition for :meth:`process_env_step`.

        During warmup (before the replay buffer can be sampled) uniform random
        actions in ``[-1, 1]`` are used. Otherwise the squashed-Gaussian actor is
        sampled with zeta-distributed noise repetition. Deterministic actions
        (``training=False``) return ``tanh(mean)``.
        """
        with torch.no_grad():
            mean, std = self.actor.mean_std(obs, training=False)

            if not training:
                actions = torch.tanh(mean)
            elif not self.storage.can_sample():
                # Warmup: uniform exploration in [-1, 1].
                actions = torch.rand_like(mean) * 2.0 - 1.0
            else:
                reinit = (self._cur_noise_count == 0) | (self._cur_noise_count >= self._cur_noise_n)
                new_noise = torch.randn_like(mean)
                new_n = self._sample_noise_repeat_n(mean.shape[0])
                self._cached_noise = torch.where(reinit.unsqueeze(-1), new_noise, self._cached_noise)
                self._cur_noise_n = torch.where(reinit, new_n, self._cur_noise_n)
                self._cur_noise_count = torch.where(
                    reinit, torch.zeros_like(self._cur_noise_count), self._cur_noise_count
                )
                actions = torch.tanh(mean + std * self._cached_noise)
                self._cur_noise_count = self._cur_noise_count + 1

        self._transition_obs = obs
        self._transition_actions = actions
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step into the replay buffer and update normalizers."""
        if self._transition_obs is None or self._transition_actions is None:
            raise RuntimeError("process_env_step() called before act(); no transition to record.")

        # Update observation normalizers from the observation the policy consumed.
        self.actor.update_normalization(self._transition_obs)
        self.critic.update_normalization(self._transition_obs)

        # Reconstruct terminated/truncated (fail loud on missing time_outs).
        if self.finite_horizon:
            truncated = torch.zeros_like(dones, dtype=torch.bool)
        else:
            if "time_outs" not in extras:
                raise KeyError(
                    "FlashSAC.process_env_step: infinite-horizon env must provide extras['time_outs'] to distinguish "
                    "truncations from terminations. Set the environment's finite-horizon flag or supply 'time_outs'."
                )
            truncated = extras["time_outs"].to(self.device).bool()
        terminated = dones.to(self.device).bool() & ~truncated

        rewards = rewards.to(self.device)
        if self.normalize_reward:
            self.reward_normalizer.update_reward_stats(rewards, terminated.float(), truncated.float())

        self.storage.add(
            observations=self._transition_obs,
            actions=self._transition_actions,
            rewards=rewards,
            terminated=terminated.float(),
            truncated=truncated.float(),
            next_observations=obs,
        )
        self._transition_obs = None
        self._transition_actions = None

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op: off-policy FlashSAC does not use Monte-Carlo/GAE return targets."""
        del obs

    # -- optimization --------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Sample a batch and run one (delayed-actor) FlashSAC gradient step."""
        if not self.storage.can_sample():
            return {}

        batch = self.storage.sample()
        obs = batch.observations
        next_obs = batch.next_observations
        actions = batch.actions
        rewards = batch.rewards
        terminated = batch.terminated
        if self.normalize_reward:
            rewards = self.reward_normalizer.normalize_rewards(rewards)

        do_actor_update = self._update_step % self.actor_update_period == 0

        loss_dict: dict[str, float] = {}
        if do_actor_update:
            actor_loss, entropy, mean_action = self._update_actor(obs, next_obs, actions)
            temperature_value, temperature_loss = self._update_temperature(entropy)
            loss_dict.update({
                "actor": actor_loss,
                "entropy": entropy,
                "mean_action": mean_action,
                "temperature": temperature_value,
                "temperature_loss": temperature_loss,
            })

        critic_loss = self._update_critic(obs, next_obs, actions, rewards, terminated)
        self._update_target()
        self._update_step += 1

        loss_dict["critic"] = critic_loss
        loss_dict["learning_rate"] = self.actor_optimizer.param_groups[0]["lr"]
        return loss_dict

    def _update_actor(self, obs: TensorDict, next_obs: TensorDict, actions: torch.Tensor) -> tuple[float, float, float]:
        """Update the actor (and return actor loss, entropy, mean action)."""
        obs_all: TensorDict = torch.cat([obs, next_obs], dim=0)  # type: ignore[assignment]
        actions_all, log_probs_all = self.actor.sample(obs_all, training=True)
        cur_actions = torch.chunk(actions_all, 2, dim=0)[0]
        log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

        # Freeze critic grads while differentiating through it.
        self.critic.net.requires_grad_(False)
        qs, _ = self.critic(obs, cur_actions, training=False)
        q = torch.minimum(qs[0], qs[1])
        self.critic.net.requires_grad_(True)

        temp_value = self.temperature().detach()
        actor_loss = (log_probs * temp_value - q).mean()
        if self.actor_bc_alpha > 0:
            # Behavior-cloning regularization (https://arxiv.org/abs/2306.02451).
            q_abs = torch.abs(q).mean().detach()
            bc_loss = ((cur_actions - actions) ** 2).mean()
            actor_loss = actor_loss + self.actor_bc_alpha * q_abs * bc_loss

        entropy = -log_probs.mean()
        mean_action = cur_actions.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()
        self.actor.normalize_parameters()

        return actor_loss.item(), entropy.item(), mean_action.item()

    def _update_temperature(self, entropy: float) -> tuple[float, float]:
        """Update the entropy temperature toward the target entropy."""
        temperature_value = self.temperature()
        temperature_loss = temperature_value * (entropy - self.temp_target_entropy)
        temperature_loss = temperature_loss.mean()

        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature_scheduler.step()

        return float(temperature_value.item()), float(temperature_loss.item())

    def _update_critic(
        self,
        obs: TensorDict,
        next_obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
    ) -> float:
        """Update the distributional double critic against the categorical TD target."""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs, training=False)
            next_actions = next_actions.clone()
            next_log_probs = next_log_probs.clone()
            temp_value = self.temperature()
            next_actor_entropy = temp_value * next_log_probs

            obs_all: TensorDict = torch.cat([obs, next_obs], dim=0)  # type: ignore[assignment]
            act_all = torch.cat([actions, next_actions], dim=0)
            qs_all, info_all = self.target_critic(obs_all, act_all, training=True)
            next_qs = qs_all.chunk(2, dim=1)[1]
            next_q_log_probs = info_all["log_prob"].chunk(2, dim=1)[1]
            next_q_log_probs = self._select_min_q_log_probs(next_qs, next_q_log_probs)

            target_probs = self._compute_categorical_td_target(
                target_log_probs=next_q_log_probs,
                reward=rewards,
                done=terminated,
                actor_entropy=next_actor_entropy,
                gamma=self.gamma**self.n_step,
                num_bins=self.num_bins,
                min_v=self.min_v,
                max_v=self.max_v,
            )

        pred_qs_all, pred_info = self.critic(obs_all, act_all, training=True)
        del pred_qs_all
        pred_log_probs = torch.chunk(pred_info["log_prob"], 2, dim=1)[0]
        ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)
        critic_loss = ce_loss.mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        self.critic.normalize_parameters()

        return critic_loss.item()

    @torch.no_grad()
    def _update_target(self) -> None:
        """Polyak (EMA) update of the target critic toward the online critic."""
        tau = self.critic_target_update_tau
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.lerp_(source_param.data, tau)

    @staticmethod
    def _select_min_q_log_probs(next_qs: torch.Tensor, next_q_log_probs: torch.Tensor) -> torch.Tensor:
        """Select the per-bin log-probs of the minimum-Q critic member."""
        num_bins = next_q_log_probs.shape[-1]
        min_indices = next_qs.argmin(dim=0)  # (B,)
        selected = torch.gather(
            next_q_log_probs,
            dim=0,
            index=min_indices[None, :, None].expand(1, -1, num_bins),
        )[0]
        return selected

    @staticmethod
    def _compute_categorical_td_target(
        target_log_probs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        actor_entropy: torch.Tensor,
        gamma: float,
        num_bins: int,
        min_v: float,
        max_v: float,
    ) -> torch.Tensor:
        """Project the entropy-augmented n-step TD target onto the categorical support."""
        batch_size = reward.shape[0]
        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)
        actor_entropy = actor_entropy.reshape(-1, 1)

        bin_width = (max_v - min_v) / (num_bins - 1)
        bin_values = torch.linspace(
            min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
        ).view(1, -1)

        target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
        target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

        b = (target_bin_values - min_v) / bin_width
        lower = torch.floor(b).long()
        upper = torch.clamp(lower + 1, 0, num_bins - 1)
        frac = b - lower.float()

        target_probs_exp = target_log_probs.exp()
        m_l = target_probs_exp * (1.0 - frac)
        m_u = target_probs_exp * frac

        target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)
        target_probs.scatter_add_(1, lower, m_l)
        target_probs.scatter_add_(1, upper, m_u)
        return target_probs

    # -- mode / export / persistence ----------------------------------------

    def train_mode(self) -> None:
        """Set train mode on all learnable modules."""
        self.actor.train()
        self.critic.train()
        self.target_critic.train()
        self.temperature.train()

    def eval_mode(self) -> None:
        """Set evaluation mode on all learnable modules."""
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()
        self.temperature.eval()

    def get_policy(self) -> FlashSACActorModel:
        """Return the actor model used for inference and export."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile the models (unsupported in v1; only ``None`` is accepted)."""
        if mode is not None:
            raise NotImplementedError(
                f"torch_compile_mode='{mode}' is not supported for FlashSAC v1 (eager-only). Set it to None."
            )

    def save(self) -> dict:
        """Return a dict of all model, optimizer, and scheduler states for saving."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "target_critic_state_dict": self._raw_target_critic.state_dict(),
            "temperature_state_dict": self.temperature.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "temperature_optimizer_state_dict": self.temperature_optimizer.state_dict(),
            "actor_scheduler_state_dict": self.actor_scheduler.state_dict(),
            "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
            "temperature_scheduler_state_dict": self.temperature_scheduler.state_dict(),
            "reward_normalizer_state_dict": self.reward_normalizer.state_dict(),
            "update_step": self._update_step,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models/states from a saved dict; return whether to restore iteration."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True, "reward_normalizer": True}

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self._raw_target_critic.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
            self.temperature.load_state_dict(loaded_dict["temperature_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.temperature_optimizer.load_state_dict(loaded_dict["temperature_optimizer_state_dict"])
            self.actor_scheduler.load_state_dict(loaded_dict["actor_scheduler_state_dict"])
            self.critic_scheduler.load_state_dict(loaded_dict["critic_scheduler_state_dict"])
            self.temperature_scheduler.load_state_dict(loaded_dict["temperature_scheduler_state_dict"])
            self._update_step = loaded_dict["update_step"]
        if load_cfg.get("reward_normalizer") and "reward_normalizer_state_dict" in loaded_dict:
            self.reward_normalizer.load_state_dict(loaded_dict["reward_normalizer_state_dict"])
        return load_cfg.get("iteration", False)

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs (unsupported: single-GPU only)."""
        raise NotImplementedError("Multi-GPU training is not supported for FlashSAC in this version.")

    def reduce_parameters(self) -> None:
        """Average gradients across GPUs (unsupported: single-GPU only)."""
        raise NotImplementedError("Multi-GPU training is not supported for FlashSAC in this version.")

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FlashSAC:
        """Construct the FlashSAC algorithm from a runner configuration dict (fail-loud)."""
        _require(
            cfg, "runner cfg", "algorithm", "actor", "critic", "replay", "obs_groups", "multi_gpu", "torch_compile_mode"
        )

        alg_class: type[FlashSAC] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FlashSACActorModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[FlashSACCriticModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_sac_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", action_dim=env.num_actions, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        target_critic = critic_class(obs, cfg["obs_groups"], "critic", action_dim=env.num_actions, **cfg["critic"]).to(
            device
        )
        target_critic.load_state_dict(critic.state_dict())

        alg_cfg = cfg["algorithm"]
        _require(
            alg_cfg,
            "algorithm cfg",
            "temp_initial_value",
            "temp_target_sigma",
            "temp_target_entropy",
            "use_amp",
            "gamma",
            "n_step",
        )

        # Temperature network + auto target entropy.
        temperature = FlashSACTemperature(alg_cfg.pop("temp_initial_value")).to(device)
        sigma = alg_cfg.pop("temp_target_sigma")
        if alg_cfg["temp_target_entropy"] is None:
            alg_cfg["temp_target_entropy"] = 0.5 * env.num_actions * math.log(2 * math.pi * math.e * sigma**2)

        # Eager-first: reject AMP explicitly.
        if alg_cfg.pop("use_amp"):
            raise NotImplementedError("use_amp=True is not supported for FlashSAC v1 (eager-only). Set use_amp=False.")

        # Storage.
        storage = ReplayStorage(
            num_envs=env.num_envs,
            obs=obs,
            actions_shape=[env.num_actions],
            **cfg["replay"],
            n_step=alg_cfg["n_step"],
            gamma=alg_cfg["gamma"],
            device=device,
        )

        # Horizon flag for terminated/truncated reconstruction.
        env_cfg = getattr(env, "cfg", None)
        alg_cfg["finite_horizon"] = bool(getattr(env_cfg, "is_finite_horizon", False))

        alg: FlashSAC = alg_class(
            actor,
            critic,
            target_critic,
            temperature,
            storage,
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg["torch_compile_mode"])
        return alg
