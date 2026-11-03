# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import functools
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import Symmetry, resolve_symmetry_config
from rsl_rl.models import CNNModel, MLPModel, QMLPModel
from rsl_rl.modules import EmpiricalNormalization, SquashedGaussianDistribution
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import CudaGraphCallable, compile_model, resolve_class, resolve_obs_groups, resolve_optimizer


class SAC:
    """Soft Actor-Critic algorithm.

    Inspired by FlashSAC, this algorithm supports categorical critics, reward normalization, temporally correlated
    exploration noise, and residual actor and critic networks.

    References:
        - Haarnoja et al. "Soft Actor-Critic Algorithms and Applications." arXiv preprint arXiv:1812.05905 (2018).
        - Kim et al. "FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control."
          arXiv preprint arXiv:2604.04539 (2026).
    """

    actor: MLPModel
    """The actor model."""

    critics: nn.ModuleList
    """The ensemble of state-action critic models."""

    target_critics: nn.ModuleList
    """The ensemble of target critic models."""

    def __init__(
        self,
        actor: MLPModel,
        critics: list[QMLPModel],
        storage: ReplayBuffer,
        # Update parameters
        num_updates: int = 1,
        batch_size: int = 256,
        num_return_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_update_interval: int = 1,
        num_warmup_transitions: int = 1000,
        # Optimizer parameters
        actor_learning_rate: float = 0.0003,
        critic_learning_rate: float = 0.0003,
        alpha_learning_rate: float = 0.0003,
        max_grad_norm: float | None = None,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        # Entropy parameters
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        target_entropy_std: float | None = None,
        # Critic loss parameters
        critic_loss_type: str = "mse",
        num_bins: int = 101,
        value_range: tuple[float, float] = (-5.0, 5.0),
        # Reward normalization parameters
        reward_normalization: bool = False,
        reward_normalization_max_return: float = 5.0,
        # Exploration parameters
        exploration_type: str = "gaussian",
        noise_repeat_exponent: float = 2.0,
        max_noise_repeat_steps: int = 16,
        # Learning rate schedule parameters
        learning_rate_schedule: str = "constant",
        final_learning_rate_ratio: float = 0.5,
        num_schedule_updates: int | None = None,
        # Execution parameters
        use_mixed_precision: bool = False,
        use_cuda_graph: bool = False,
        device: str = "cpu",
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Not yet implemented features
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Multi-GPU training is not yet supported for SAC.")
        if actor.is_recurrent or any(critic.is_recurrent for critic in critics):
            raise ValueError("SAC does not yet support recurrent models.")
        if isinstance(actor, CNNModel):
            raise ValueError("SAC does not yet support CNN models.")

        # Validate configuration
        if not isinstance(actor.distribution, SquashedGaussianDistribution):
            raise ValueError("SAC requires a SquashedGaussianDistribution actor.")
        if len(critics) < 2:
            raise ValueError("SAC requires at least two critics.")
        if critic_loss_type not in ("mse", "categorical"):
            raise ValueError(f"Unsupported critic loss: {critic_loss_type}. Expected 'mse' or 'categorical'.")
        if exploration_type not in ("gaussian", "held"):
            raise ValueError(f"Unsupported exploration mode: {exploration_type}. Expected 'gaussian' or 'held'.")
        if learning_rate_schedule not in ("constant", "cosine"):
            raise ValueError("Unsupported learning rate schedule.")
        if learning_rate_schedule == "cosine" and num_schedule_updates is None:
            raise ValueError("A cosine schedule requires num_schedule_updates to be specified.")

        # Symmetry extension
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg is not None else None

        # Device-related parameters
        if use_cuda_graph and torch.device(device).type != "cuda":
            raise ValueError("CUDA graph capture requires a CUDA device.")
        self.use_cuda_graph = use_cuda_graph
        self.device = device

        # SAC components
        self.actor = actor.to(self.device)
        self.critics = nn.ModuleList(critics).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)

        # Handles to the uncompiled modules for state_dict operations and export
        self._raw_actor = self.actor
        self._raw_critics = self.critics

        # Initialize the entropy temperature and create the optimizers
        self.log_alpha = torch.tensor(math.log(init_alpha), device=self.device, requires_grad=True)
        optimizer_class = resolve_optimizer(optimizer)
        self.actor_optimizer = optimizer_class(
            self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay
        )  # type: ignore
        self.critic_optimizer = optimizer_class(
            self.critics.parameters(), lr=critic_learning_rate, weight_decay=weight_decay
        )  # type: ignore
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.learning_rate = critic_learning_rate

        # Add storage
        self.storage = storage
        self.transition = ReplayBuffer.Transition()

        # SAC parameters
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.num_return_steps = num_return_steps
        self._discounts = gamma ** torch.arange(num_return_steps + 1, dtype=torch.float, device=self.device)
        self.gamma = gamma
        self.tau = tau
        self.actor_update_interval = actor_update_interval
        self.num_warmup_transitions = num_warmup_transitions
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision

        # Entropy parameters
        self.target_entropy = -float(actor.distribution.output_dim) if target_entropy is None else target_entropy
        if target_entropy is None and target_entropy_std is not None:
            self.target_entropy = (
                0.5 * actor.distribution.output_dim * math.log(2 * math.pi * math.e * target_entropy_std**2)
            )

        # Categorical critic parameters
        self.critic_loss_type = critic_loss_type
        self.num_bins = num_bins
        self.value_range = value_range
        self._bin_values = torch.linspace(*value_range, num_bins, device=self.device)

        # Reward normalization
        self.reward_normalization = reward_normalization
        self.reward_normalization_max_return = reward_normalization_max_return
        self._return_normalizer = EmpiricalNormalization(1, eps=0.0).to(self.device)
        self._returns = torch.zeros(storage.num_envs, device=self.device)
        self._max_abs_return = torch.zeros((), device=self.device)
        self._pending_returns: list[torch.Tensor] = []

        # Exploration parameters
        self.exploration_type = exploration_type
        repeats = torch.arange(1, max_noise_repeat_steps + 1, dtype=torch.float, device=self.device)
        self._noise_repeat_cdf = torch.cumsum(repeats**-noise_repeat_exponent, dim=0)
        self._noise_repeat_cdf /= self._noise_repeat_cdf[-1].clone()
        self._noise = torch.zeros(storage.num_envs, actor.distribution.output_dim, device=self.device)
        self._noise_steps_left = torch.zeros(storage.num_envs, dtype=torch.long, device=self.device)

        # Collect batch normalization layers
        self._actor_batch_norms = _batch_norm_layers(self.actor)
        self._critic_batch_norms = _batch_norm_layers(self.critics)
        self._target_batch_norms = _batch_norm_layers(self.target_critics)

        # Create the learning rate schedulers
        self._schedulers: list[torch.optim.lr_scheduler.CosineAnnealingLR] = []
        if learning_rate_schedule == "cosine":
            for model_optimizer in (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer):
                self._schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        model_optimizer,
                        T_max=num_schedule_updates,  # type: ignore
                        eta_min=model_optimizer.param_groups[0]["lr"] * final_learning_rate_ratio,
                    )
                )

        # Initialize update counters, actor state, and pending observations
        self.num_env_steps = 0
        self.num_critic_updates = 0
        self._actor_loaded = False
        self._pending_observations: list[TensorDict] = []

        # Configure optimizer state for CUDA graph capture
        self._configure_optimizer_capture()

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Keep policy statistics available to the runner during uniform-action warm-up
        with _BatchNormMode(self._actor_batch_norms, training=False):
            actions = self.actor(obs, stochastic_output=True)

        # Sample uniform actions during warm-up
        distribution = self.actor.distribution
        if not self._actor_loaded and self.num_env_steps * self.storage.num_envs < self.num_warmup_transitions:
            uniform_actions = torch.rand(obs.batch_size[0], distribution.output_dim, device=self.device) * 2.0 - 1.0
            actions = distribution.range_offset + distribution.range_scale * uniform_actions  # type: ignore
        elif self.exploration_type == "held":
            # Resample noise and repetition counts for environments whose held noise has expired
            resample = self._noise_steps_left <= 0
            self._noise.copy_(torch.where(resample[:, None], torch.randn_like(self._noise), self._noise))
            repeats = torch.searchsorted(self._noise_repeat_cdf, torch.rand_like(self._noise_steps_left.float()))
            self._noise_steps_left.copy_(torch.where(resample, repeats + 1, self._noise_steps_left) - 1)
            actions = distribution.sample(noise=self._noise)  # type: ignore

        # Record observations and actions before env.step()
        self.transition.observations = obs
        self.transition.actions = actions
        return actions

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """Record one environment step using final observations for environments that reset."""
        # Resolve termination flags and observation groups
        done_mask = dones.reshape(-1).bool()
        time_outs = extras.get("time_outs", torch.zeros_like(done_mask)).to(self.device).reshape(-1).bool()
        critic_obs_groups = [obs_group for critic in self.critics for obs_group in critic.obs_groups]  # type: ignore
        groups = list(dict.fromkeys(self.actor.obs_groups + critic_obs_groups))

        # Replace reset observations with final observations for completed episodes
        next_obs = obs.select(*groups).clone()
        if done_mask.any():
            if extras.get("final_obs") is None:
                raise ValueError("SAC requires extras['final_obs'] on every step containing a done.")
            final_obs = extras["final_obs"]
            for key in groups:
                mask = done_mask.reshape(-1, *([1] * (next_obs[key].ndim - 1)))
                next_obs[key] = torch.where(mask, final_obs[key].to(self.device), next_obs[key])

        # Construct the transition using observations and actions recorded by act()
        observations = self.transition.observations
        self.transition.observations = observations.select(*groups)
        self.transition.rewards = rewards.reshape(-1)
        self.transition.next_observations = next_obs
        self.transition.dones = done_mask
        self.transition.terminated = done_mask & ~time_outs

        # Record the transition and retain observations for normalization
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self._pending_observations.append(observations.detach().clone())

        # Record discounted returns for reward normalization
        if self.reward_normalization:
            self._returns.mul_(self.gamma).add_(rewards.reshape(-1))
            self._max_abs_return.copy_(torch.maximum(self._max_abs_return, self._returns.abs().max()))
            self._pending_returns.append(self._returns.clone())
            self._returns.mul_(~dones.reshape(-1).bool())

        # Reset exploration and model state for completed episodes
        self._noise_steps_left.mul_(~dones.reshape(-1).bool())
        self.num_env_steps += 1
        self.actor.reset(dones)
        for critic in self.critics:
            critic.reset(dones)  # type: ignore

    def compute_returns(self, obs: TensorDict) -> None:
        """Defer return computation to replay sampling in :meth:`update` (no-op)."""
        pass

    def update(self) -> dict[str, float]:
        """Update normalizers and models from replay samples and return mean losses."""
        # Update the normalizers using newly collected observations and returns
        if self._pending_observations:
            observations = torch.cat(self._pending_observations)  # type: ignore
            self.actor.update_normalization(observations)  # type: ignore
            for critic in self.critics:
                critic.update_normalization(observations)  # type: ignore
            self._pending_observations.clear()
        if self._pending_returns:
            self._return_normalizer.update(torch.cat(self._pending_returns).unsqueeze(-1))
            self._pending_returns.clear()

        # Wait until enough transitions have been collected
        if self.storage.size < self.num_warmup_transitions or not self.storage.size:
            return {}

        # Initialize the loss accumulators
        totals: dict[str, torch.Tensor] = {}
        actor_updates = 0

        # Iterate over replay mini-batches
        for _ in range(self.num_updates):
            # Sample and augment transitions and normalize rewards if enabled
            batch = self._sample_batch()
            if self.reward_normalization:
                batch.rewards = batch.rewards / torch.maximum(
                    self._return_normalizer._std.squeeze() + 1e-8,  # type: ignore
                    self._max_abs_return / self.reward_normalization_max_return,
                )

            # Run the gradient update with the configured actor update interval
            update_actor = self.num_critic_updates % self.actor_update_interval == 0
            inputs = (
                batch.observations,
                batch.actions,
                batch.rewards,
                batch.next_observations,
                batch.bootstrap_discounts,
            )
            if self.use_cuda_graph:
                stats = self._graph_steps[update_actor](*inputs)
            else:
                stats = self._gradient_step(*inputs, update_actor=update_actor)
            self.num_critic_updates += 1

            # Update the learning rate schedules
            for scheduler in self._schedulers:
                if scheduler.last_epoch < scheduler.T_max:
                    scheduler.step()
            self.learning_rate = float(self.critic_optimizer.param_groups[0]["lr"])

            # Store the losses
            actor_updates += int(update_actor)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value

        # Divide the losses by the number of updates
        loss_dict: dict[str, float] = {}
        for key, value in totals.items():
            num_updates = actor_updates if key in ("actor", "alpha", "entropy", "symmetry") else self.num_updates
            loss_dict[key] = (value / num_updates).item()
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critics.train()
        self.target_critics.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critics.eval()
        self.target_critics.eval()

    def save(self) -> dict:
        """Return a dict of models, optimizers, temperature, and update counters for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critics.state_dict(),
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
            saved_dict["return_normalizer_state_dict"] = self._return_normalizer.state_dict()
            saved_dict["max_abs_return"] = self._max_abs_return.clone()
        return saved_dict

    def get_default_load_cfg(self) -> dict:
        """Return the default selection for resuming this algorithm's own checkpoints."""
        return {"actor": True, "critic": True, "optimizer": True, "iteration": True}

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict.

        .. note::
            Loading an actor disables uniform action sampling during warm-up.
        """
        # Resolve the default load configuration
        if load_cfg is None:
            load_cfg = self.get_default_load_cfg()

        # Load the specified models and temperature
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
            self._actor_loaded = True
        if load_cfg.get("critic"):
            self._raw_critics.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self.target_critics.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
            if self.reward_normalization:
                self._return_normalizer.load_state_dict(loaded_dict["return_normalizer_state_dict"], strict=strict)
                self._max_abs_return.copy_(loaded_dict["max_abs_return"])
            with torch.no_grad():
                self.log_alpha.copy_(loaded_dict["log_alpha"])

        # Load the optimizer states and learning rate schedules
        if load_cfg.get("optimizer"):
            for name in ("actor", "critic", "alpha"):
                getattr(self, f"{name}_optimizer").load_state_dict(loaded_dict[f"{name}_optimizer_state_dict"])
            self.learning_rate = float(self.critic_optimizer.param_groups[0]["lr"])
            for scheduler, state in zip(self._schedulers, loaded_dict.get("scheduler_state_dicts", [])):
                scheduler.load_state_dict(state)

        # Restore the update counters
        if load_cfg.get("iteration"):
            self.num_env_steps = loaded_dict["num_env_steps"]
            self.num_critic_updates = loaded_dict["num_critic_updates"]

        # Reconfigure optimizer for CUDA graph capture
        self._configure_optimizer_capture()

        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critics with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critics = nn.ModuleList([compile_model(critic, mode) for critic in self._raw_critics])

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAC:
        """Construct the SAC algorithm."""
        # Resolve class callables and configs
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        # Resolve symmetry configuration if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the actor and critics
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        num_critics = alg_cfg.pop("num_critics", 2)  # Number of critics to use in the ensemble
        critic_output_dim = (
            alg_cfg.get("num_bins", 101) if alg_cfg.get("critic_loss_type", "mse") == "categorical" else 1
        )
        critics = [
            critic_class(obs, cfg["obs_groups"], "critic", critic_output_dim, env.num_actions, **critic_cfg).to(device)
            for _ in range(num_critics)
        ]
        print(f"Critic Models: {critics[0]} (x{num_critics})")

        # Initialize the storage
        storage = ReplayBuffer(
            env.num_envs,
            alg_cfg.pop("replay_buffer_capacity", 100_000),
            min_steps_per_env=max(
                alg_cfg.get("num_return_steps", 1),
                math.ceil(alg_cfg.get("num_warmup_transitions", 1000) / env.num_envs),
            ),
            max_memory_fraction=alg_cfg.pop("max_replay_memory_fraction", None),
            device=device,
        )

        # Initialize the algorithm
        alg = alg_class(actor, critics, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def _sample_batch(self) -> _Batch:
        """Build n-step returns, bootstrapping from the next state of the last valid transition."""
        # Accumulate discounted rewards over valid replay transitions
        replay_batch = self.storage.sample(self.batch_size, length=self.num_return_steps)
        rewards = (replay_batch.window["rewards"] * self._discounts[: self.num_return_steps] * replay_batch.valid).sum(
            1
        )
        discounts = self._discounts[replay_batch.valid.sum(1)] * ~replay_batch.terminated

        # Construct the batch
        batch = _Batch(
            observations=replay_batch.observations,
            actions=replay_batch.actions,
            rewards=rewards,
            next_observations=replay_batch.next_observations,
            bootstrap_discounts=discounts,
        )

        # Apply symmetry data augmentation if enabled
        if self.symmetry is not None and self.symmetry.use_data_augmentation:
            # Apply the same augmentation to current and next observations
            augment = self.symmetry.data_augmentation_func
            batch.observations, batch.actions = augment(
                env=self.symmetry.env, obs=batch.observations, actions=batch.actions
            )
            batch.next_observations, _ = augment(env=self.symmetry.env, obs=batch.next_observations, actions=None)
            # Repeat scalar targets to match the augmented observations and actions
            num_aug = batch.actions.shape[0] // self.batch_size
            batch.rewards = batch.rewards.repeat(num_aug)
            batch.bootstrap_discounts = batch.bootstrap_discounts.repeat(num_aug)

        return batch

    def _gradient_step(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: TensorDict,
        bootstrap_discounts: torch.Tensor,
        update_actor: bool,
    ) -> dict[str, torch.Tensor]:
        """Perform one fixed-shape optimizer step, suitable for lazy CUDA capture."""
        batch = _Batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            bootstrap_discounts=bootstrap_discounts,
        )

        # Update the actor and temperature before the critics
        stats = self._update_actor_and_alpha(batch) if update_actor else {}
        stats.update(self._update_critics(batch))
        self._update_target_critics()
        return stats

    def _update_actor_and_alpha(self, batch: _Batch) -> dict[str, torch.Tensor]:
        """Run one gradient step for the actor and entropy temperature."""
        with torch.autocast(
            device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
        ):
            # Sample actions using joint batches when batch normalization is enabled
            if len(self._actor_batch_norms) > 0:
                with _BatchNormMode(self._actor_batch_norms, training=True):
                    actions, log_probs = self._sample_actions(torch.cat([batch.observations, batch.next_observations]))  # type: ignore
                actions = actions[: batch.actions.shape[0]]
                log_probs = log_probs[: batch.actions.shape[0]]
            else:
                actions, log_probs = self._sample_actions(batch.observations)

            # Evaluate policy actions without accumulating critic parameter gradients
            self.critics.requires_grad_(False)
            try:
                with _BatchNormMode(self._critic_batch_norms, training=False):
                    q_values = torch.stack([
                        self._expected_value(critic(batch.observations, self._normalize_actions(actions)))
                        for critic in self.critics
                    ])
            finally:
                self.critics.requires_grad_(True)

            # Compute the actor objective
            q_value = q_values.min(dim=0).values
            actor_loss = (self.log_alpha.detach().exp() * log_probs - q_value).mean()

            # Compute the symmetry loss and add it to the actor loss if enabled
            symmetry_loss = None
            if self.symmetry is not None:
                symmetry_loss = self.symmetry.compute_loss(self.actor, copy.copy(batch), self.batch_size)  # type: ignore
                if self.symmetry.use_mirror_loss:
                    actor_loss = actor_loss + self.symmetry.mirror_loss_coeff * symmetry_loss

        # Apply the gradients for the actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        _normalize_parameters(self.actor)

        # Update the entropy temperature
        entropy = -log_probs.detach().float().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Store the losses and sampled entropy
        stats = {"actor": actor_loss.detach(), "alpha": alpha_loss.detach(), "entropy": entropy}
        if symmetry_loss is not None:
            stats["symmetry"] = symmetry_loss.detach()
        return stats

    def _update_critics(self, batch: _Batch) -> dict[str, torch.Tensor]:
        """Run one gradient step for the critics."""
        with torch.autocast(
            device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
        ):
            # Compute the critic targets without gradients
            with torch.no_grad():
                with _BatchNormMode(self._actor_batch_norms, training=False):
                    next_actions, next_log_probs = self._sample_actions(batch.next_observations)
                if len(self._critic_batch_norms) > 0:
                    # Update target batch statistics using current and next observations
                    target_obs = torch.cat([batch.observations, batch.next_observations])
                    target_actions = torch.cat([
                        self._normalize_actions(batch.actions),
                        self._normalize_actions(next_actions),
                    ])
                    with _BatchNormMode(self._target_batch_norms, training=True):
                        next_outputs = [
                            critic(target_obs, target_actions)[batch.actions.shape[0] :]
                            for critic in self.target_critics
                        ]
                else:
                    next_outputs = [
                        critic(batch.next_observations, self._normalize_actions(next_actions))
                        for critic in self.target_critics
                    ]
                target = self._compute_target(next_outputs, batch, self.log_alpha.detach().exp() * next_log_probs)

            # Evaluate the current state-action values
            if len(self._critic_batch_norms) > 0:
                with _BatchNormMode(self._critic_batch_norms, training=True):
                    outputs = [critic(target_obs, target_actions)[: batch.actions.shape[0]] for critic in self.critics]
            else:
                outputs = [
                    critic(batch.observations, self._normalize_actions(batch.actions)) for critic in self.critics
                ]

            # Compute scalar or categorical critic losses
            if self.critic_loss_type == "mse":
                losses = [(output.squeeze(-1) - target).square().mean() for output in outputs]
            else:
                losses = [
                    -(target * torch.log_softmax(output.float(), dim=-1)).sum(dim=-1).mean() for output in outputs
                ]
            critic_loss = torch.stack(losses).mean()

        # Apply the gradients for the critics
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        _normalize_parameters(self.critics)

        # Store the critic loss and mean Q-value
        with torch.no_grad():
            q_values = torch.stack([self._expected_value(output) for output in outputs])
            stats = {"critic": critic_loss.detach(), "q_value": q_values.mean()}
        return stats

    @torch.no_grad()
    def _update_target_critics(self) -> None:
        """Interpolate target parameters and copy observation normalizers, retaining target batch statistics."""
        # Interpolate the target critic parameters
        torch._foreach_lerp_(list(self.target_critics.parameters()), list(self.critics.parameters()), self.tau)  # type: ignore

        # Copy observation normalizers without overwriting target batch statistics
        for target, source in zip(self.target_critics, self.critics):
            for target_buffer, source_buffer in zip(target.obs_normalizer.buffers(), source.obs_normalizer.buffers()):  # type: ignore
                target_buffer.copy_(source_buffer)

    def _compute_target(
        self, next_outputs: list[torch.Tensor], batch: _Batch, entropy_bonus: torch.Tensor
    ) -> torch.Tensor:
        """Compute the shared critic target (values for scalar critics, probabilities for categorical critics)."""
        # Select the critic with the lowest expected value
        next_values = [self._expected_value(output) for output in next_outputs]
        critic_indices = torch.stack(next_values).argmin(dim=0)
        next_output = torch.stack(next_outputs).gather(
            0, critic_indices.view(1, -1, 1).expand(1, -1, next_outputs[0].shape[-1])
        )[0]

        # Bootstrap scalar values or project categorical distributions onto the fixed support
        if self.critic_loss_type == "mse":
            return batch.rewards + batch.bootstrap_discounts * (next_output.squeeze(-1) - entropy_bonus)
        return self._project(
            torch.softmax(next_output.float(), dim=-1), batch.rewards, batch.bootstrap_discounts, entropy_bonus
        )

    def _project(
        self, probs: torch.Tensor, rewards: torch.Tensor, discounts: torch.Tensor, entropy_bonus: torch.Tensor
    ) -> torch.Tensor:
        """Project the shifted and scaled categorical distribution onto the fixed support."""
        # Map the shifted categorical support to fractional bin positions
        v_min, v_max = self.value_range
        delta = (v_max - v_min) / (self.num_bins - 1)
        shifted = rewards.unsqueeze(1) + discounts.unsqueeze(1) * (self._bin_values - entropy_bonus.unsqueeze(1))
        positions = (shifted.clamp(v_min, v_max) - v_min) / delta

        # Distribute each probability between its neighboring bins
        lower = positions.floor().long().clamp(0, self.num_bins - 1)
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = positions - lower.float()
        target = torch.zeros_like(probs)
        target.scatter_add_(1, lower, probs * (1.0 - upper_weight))
        target.scatter_add_(1, upper, probs * upper_weight)
        return target

    def _expected_value(self, output: torch.Tensor) -> torch.Tensor:
        """Return the Q-value of a critic output."""
        if self.critic_loss_type == "mse":
            return output.squeeze(-1).float()
        return (torch.softmax(output.float(), dim=-1) * self._bin_values).sum(dim=-1)

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Express fixed-range actions in normalized coordinates for critic inputs."""
        distribution = self.actor.distribution
        return (actions - distribution.range_offset) / distribution.range_scale  # type: ignore

    def _sample_actions(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actor actions and corrected log probabilities."""
        actions = self.actor(obs, stochastic_output=True)
        return actions, self.actor.get_output_log_prob(actions)

    def _configure_optimizer_capture(self) -> None:
        """Configure optimizers and static update functions for CUDA graph capture."""
        # Configure capturable learning rates and optimizer step counters
        for optimizer in (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer):
            for param_group in optimizer.param_groups:
                if self.use_cuda_graph and "capturable" not in param_group:
                    raise ValueError("CUDA graphs require a capturable optimizer such as Adam or AdamW.")
                if "capturable" in param_group:
                    param_group["capturable"] = self.use_cuda_graph
                if self.use_cuda_graph:
                    param_group["lr"] = torch.as_tensor(param_group["lr"], device=self.device)
                else:
                    param_group["lr"] = float(param_group["lr"])
            for state in optimizer.state.values():
                if "step" in state:
                    state["step"] = state["step"].to(self.device if self.use_cuda_graph else "cpu")

        # Create separate graphs for updates with and without an actor step
        if self.use_cuda_graph:
            self._graph_steps: dict[bool, CudaGraphCallable] = {
                update_actor: CudaGraphCallable(
                    functools.partial(self._gradient_step, update_actor=update_actor), self.device
                )
                for update_actor in (True, False)
            }


@dataclass
class _Batch:
    """A batch of replay transitions with n-step returns and bootstrap discounts."""

    observations: TensorDict
    """Observations at the start of each sampled transition."""

    actions: torch.Tensor
    """Actions taken at the start of each sampled transition."""

    rewards: torch.Tensor
    """Discounted n-step returns over the valid replay window."""

    next_observations: TensorDict
    """Observations after the last valid transition in each replay window."""

    bootstrap_discounts: torch.Tensor
    """Discount factors for bootstrapping, with zero values for terminated episodes."""


class _BatchNormMode:
    """Context manager that temporarily sets the mode of batch normalization layers."""

    def __init__(self, layers: list[nn.Module], training: bool) -> None:
        """Initialize the temporary training mode for the given layers."""
        self.layers = layers
        self.training = training

    def __enter__(self) -> None:
        """Store the previous training modes and apply the temporary mode."""
        self.previous = [layer.training for layer in self.layers]
        for layer in self.layers:
            layer.training = self.training

    def __exit__(self, *args: object) -> None:
        """Restore the previous training modes."""
        for layer, training in zip(self.layers, self.previous):
            layer.training = training


def _batch_norm_layers(module: nn.Module) -> list[nn.Module]:
    """Return the batch normalization layers of a module."""
    return [layer for layer in module.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm)]


def _normalize_parameters(module: nn.Module) -> None:
    """Apply weight normalization of networks that define it after an optimizer step."""
    for submodule in module.modules():
        if submodule is not module and hasattr(submodule, "normalize_parameters"):
            submodule.normalize_parameters()  # type: ignore
