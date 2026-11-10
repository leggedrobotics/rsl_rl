# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import functools
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from collections.abc import Callable
from dataclasses import dataclass
from tensordict import TensorDict
from typing import Any

from rsl_rl.env import VecEnv
from rsl_rl.extensions import Symmetry, resolve_symmetry_config
from rsl_rl.models import MLPModel, QCNNModel, QMLPModel
from rsl_rl.modules import EmpiricalNormalization, SquashedGaussianDistribution
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import (
    CudaGraphCallable,
    compile_model,
    reduce_gradients_in_buckets,
    resolve_class,
    resolve_obs_groups,
    resolve_optimizer,
)


class SAC:
    """Soft Actor-Critic algorithm.

    Inspired by FlashSAC, this implementation supports categorical critics, reward normalization, temporally correlated
    exploration noise, and batch-normalized residual actor and critic networks. It also supports the library's symmetry
    extension and CUDA graph updates.

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
        critics: list[QMLPModel | QCNNModel],
        storage: ReplayBuffer,
        # Update parameters
        num_updates: int = 16,
        batch_size: int = 2048,
        num_return_steps: int = 3,
        gamma: float = 0.98,
        tau: float = 0.01,
        actor_update_interval: int = 2,
        num_warmup_transitions: int = 100_000,
        # Optimizer parameters
        actor_learning_rate: float = 0.0001,
        critic_learning_rate: float = 0.0001,
        alpha_learning_rate: float = 0.0001,
        max_grad_norm: float | None = None,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        # Entropy parameters
        init_alpha: float = 0.01,
        target_entropy: float | None = None,
        target_entropy_std: float | None = 0.15,
        # Critic loss parameters
        critic_loss_type: str = "categorical",
        num_bins: int = 101,
        value_range: tuple[float, float] = (-5.0, 5.0),
        # Reward normalization parameters
        reward_normalization: bool = True,
        reward_normalization_max_return: float = 5.0,
        # Exploration parameters
        exploration_type: str = "held",
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
        grad_reduce_bucket_mb: float = 25,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Not yet implemented features
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb
        if not math.isfinite(grad_reduce_bucket_mb) or grad_reduce_bucket_mb <= 0:
            raise ValueError("grad_reduce_bucket_mb must be positive and finite.")
        if self.is_multi_gpu:
            if not dist.is_initialized():
                raise ValueError("Distributed SAC requires an initialized process group.")
            if dist.get_world_size() != self.gpu_world_size or dist.get_rank() != self.gpu_global_rank:
                raise ValueError("SAC multi_gpu_cfg does not match the initialized process group.")
            if use_cuda_graph and dist.get_backend() != "nccl":
                raise ValueError("Distributed SAC can only capture CUDA graphs with the NCCL backend.")
        if actor.is_recurrent or any(critic.is_recurrent for critic in critics):
            raise ValueError("SAC does not yet support recurrent models.")

        # Validate configuration
        if not isinstance(actor.distribution, SquashedGaussianDistribution):
            raise ValueError("SAC requires a SquashedGaussianDistribution actor.")
        if len(critics) < 2:
            raise ValueError("SAC requires at least two critics.")
        parameter_ids: set[int] = set()
        for model in (actor, *critics):
            model_ids = {id(parameter) for parameter in model.parameters()}
            if parameter_ids & model_ids:
                raise ValueError(
                    "SAC requires separate parameters for the actor and each critic, including CNN encoders."
                )
            parameter_ids.update(model_ids)
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

        # Learning rate schedule parameters
        self.learning_rate_schedule = learning_rate_schedule
        self.final_learning_rate_ratio = final_learning_rate_ratio
        self.num_schedule_updates = num_schedule_updates
        self._base_learning_rates = (actor_learning_rate, critic_learning_rate, alpha_learning_rate)

        # Observation groups stored in replay and flags for final observations that are not finite
        self._obs_groups = list(
            dict.fromkeys(
                group
                for model in (self.actor, *self.critics)
                for group in model.obs_groups + getattr(model, "obs_groups_2d", [])
            )
        )
        self._nonfinite_final_obs = torch.zeros(len(self._obs_groups), dtype=torch.bool, device=self.device)
        self._missing_final_obs = torch.zeros((), dtype=torch.bool, device=self.device)

        # Initialize update counters, actor state, and collection graphs
        self.num_env_steps = 0
        self.num_critic_updates = 0
        self._num_critic_updates_tensor = torch.zeros((), device=self.device)
        self._num_steps_since_update = 0
        self._actor_loaded = False
        self._collection_graphs: dict[tuple, CudaGraphCallable] = {}

        # Configure optimizer state for CUDA graph capture
        self._configure_optimizer_capture()
        self.broadcast_parameters()

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Sample uniform actions during warm-up unless an actor was loaded
        uniform = not self._actor_loaded and self.num_env_steps * self.storage.num_envs < self.num_warmup_transitions
        actions = self._run_collection_step(("act", uniform), functools.partial(self._act, uniform=uniform), obs)

        # Record observations and actions before env.step()
        self.transition.observations = obs
        self.transition.actions = actions["actions"]
        return actions["actions"]

    @torch.no_grad()
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> dict[str, torch.Tensor]:
        """Record one environment step and return per-environment values that the logger sums over each episode.

        Final observations are used as next observations for environments that reset.
        """
        # Resolve termination flags and observation groups
        done_mask = dones.reshape(-1).bool()
        time_outs = extras.get("time_outs", torch.zeros_like(done_mask)).to(self.device).reshape(-1).bool()
        groups = self._obs_groups
        final_obs = extras.get("final_obs")
        if final_obs is None:
            self._missing_final_obs.logical_or_(done_mask.any())
            final_obs = obs

        # Record the transition and update the episode state
        self._run_collection_step(
            ("process_env_step",),
            self._process_env_step,
            self.transition.observations.select(*groups),  # type: ignore
            self.transition.actions,
            obs.select(*groups),
            rewards.reshape(-1),
            done_mask,
            time_outs,
            TensorDict({key: final_obs[key] for key in groups}, batch_size=obs.batch_size).to(self.device),
        )
        self.storage.advance()
        self.transition.clear()
        self.num_env_steps += 1
        self._num_steps_since_update += 1

        # Reset model state for completed episodes
        self.actor.reset(dones)
        for critic in self.critics:
            critic.reset(dones)  # type: ignore

        return {"Training/mean_reward": rewards}

    def compute_returns(self, obs: TensorDict) -> None:
        """Defer return computation to replay sampling in :meth:`update` (no-op)."""
        pass

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Update normalizers and models from replay samples and return mean losses and statistics."""
        # Reject training on missing or nonfinite final observations, consistently across ranks
        errors = torch.cat([self._missing_final_obs.view(1), self._nonfinite_final_obs]).long()
        if self.is_multi_gpu:
            dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        missing, *nonfinite = errors.tolist()
        if missing:
            raise ValueError("SAC requires extras['final_obs'] on every step containing a done (on every rank).")
        if any(nonfinite):
            groups = [group for group, flag in zip(self._obs_groups, nonfinite) if flag]
            raise ValueError(f"SAC received nonfinite final_obs in observation groups {groups} on at least one rank.")

        # Update the observation normalizers using the transitions collected since the last update
        if self._num_steps_since_update:
            observations = self.storage.latest_observations(self._num_steps_since_update)
            self.actor.update_normalization(observations)  # type: ignore
            for critic in self.critics:
                critic.update_normalization(observations)  # type: ignore
            self._num_steps_since_update = 0
        if self.is_multi_gpu:
            dist.all_reduce(self._max_abs_return, op=dist.ReduceOp.MAX)

        # Wait until enough transitions have been collected
        ready = self.storage.size >= self.num_warmup_transitions and self.storage.size > 0
        if self.is_multi_gpu:
            ready_tensor = torch.tensor(ready, device=self.device, dtype=torch.long)
            dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
            ready = bool(ready_tensor)
        if not ready:
            return {}, self._get_learning_rate_stats() | _to_floats(self._get_model_stats())

        # Run all gradient updates of this iteration, captured as a single CUDA graph if enabled
        phase = self.num_critic_updates % self.actor_update_interval
        if self.use_cuda_graph:
            if phase not in self._graph_updates:
                self._graph_updates[phase] = CudaGraphCallable(
                    functools.partial(self._run_updates, phase=phase), self.device
                )
            totals = self._graph_updates[phase]()
        else:
            totals = self._run_updates(phase=phase)
        self.num_critic_updates += self.num_updates
        self.learning_rate = self._base_learning_rates[1] * self._learning_rate_factor(self.num_critic_updates)

        # Divide the losses by the number of updates and move entropy and Q-values to the statistics
        actor_updates = sum((phase + i) % self.actor_update_interval == 0 for i in range(self.num_updates))
        actor_keys = ("actor", "alpha", "entropy", "symmetry")
        losses = {
            key: value.double() / (actor_updates if key in actor_keys else self.num_updates)
            for key, value in totals.items()
        }
        stats = self._get_model_stats()
        if "entropy" in losses:
            stats["Model/mean_policy_entropy"] = losses.pop("entropy")
        stats["Model/mean_critic_q_value"] = losses.pop("q_value")

        # Synchronize with the device once for all losses and statistics
        values = _to_floats({**losses, **stats})
        stats_dict = self._get_learning_rate_stats() | {key: values[key] for key in stats}
        return {key: values[key] for key in losses}, stats_dict

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

        # Restore the update counters
        if load_cfg.get("iteration"):
            self.num_env_steps = loaded_dict["num_env_steps"]
            self.num_critic_updates = loaded_dict["num_critic_updates"]
            self._num_critic_updates_tensor.fill_(self.num_critic_updates)
        self.learning_rate = self._base_learning_rates[1] * self._learning_rate_factor(self.num_critic_updates)

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
        if actor_class.is_recurrent or critic_class.is_recurrent:
            raise ValueError("SAC does not yet support recurrent models.")
        if not issubclass(critic_class, (QMLPModel, QCNNModel)):  # type: ignore
            raise ValueError("SAC requires QMLPModel or QCNNModel critics.")

        # Resolve observation groups
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        # Resolve symmetry configuration if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the actor and critics
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        num_critics = alg_cfg.pop("num_critics", 2)  # Number of critics to use in the ensemble
        critic_output_dim = (
            alg_cfg.get("num_bins", 101) if alg_cfg.get("critic_loss_type", "categorical") == "categorical" else 1
        )
        critics = [
            critic_class(obs, cfg["obs_groups"], "critic", critic_output_dim, env.num_actions, **critic_cfg).to(device)
            for _ in range(num_critics)
        ]
        print(f"Critic Models: {critics[0]} (x{num_critics})")

        # Initialize the storage
        storage = ReplayBuffer(
            env.num_envs,
            alg_cfg.pop("replay_buffer_capacity", 10_000_000),
            min_steps_per_env=max(
                alg_cfg.get("num_return_steps", 3),
                math.ceil(alg_cfg.get("num_warmup_transitions", 100_000) / env.num_envs),
            ),
            max_memory_fraction=alg_cfg.pop("max_replay_memory_fraction", None),
            device=device,
        )

        # Initialize the algorithm
        alg = alg_class(actor, critics, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    @torch.no_grad()
    def broadcast_parameters(self) -> None:
        """Synchronize online models, targets, normalizers, temperature and update counters from rank zero."""
        if not self.is_multi_gpu:
            return
        for model in (self._raw_actor, self._raw_critics, self.target_critics, self._return_normalizer):
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)
        dist.broadcast(self.log_alpha, src=0)
        dist.broadcast(self._max_abs_return, src=0)
        counters = torch.tensor(
            [self.num_env_steps, self.num_critic_updates, int(self._actor_loaded)], device=self.device
        )
        dist.broadcast(counters, src=0)
        self.num_env_steps, self.num_critic_updates, actor_loaded = counters.tolist()
        self._num_critic_updates_tensor.fill_(self.num_critic_updates)
        self._actor_loaded = bool(actor_loaded)

    def reduce_parameters(self, optimizer: torch.optim.Optimizer) -> None:
        """Average optimizer gradients across ranks before clipping and stepping."""
        if self.is_multi_gpu:
            parameters = (parameter for group in optimizer.param_groups for parameter in group["params"])
            reduce_gradients_in_buckets(parameters, self.gpu_world_size, self.grad_reduce_bucket_mb)

    @torch.no_grad()
    def _synchronize_batch_norm_buffers(self) -> None:
        """Average local running statistics; forward passes still use local minibatch statistics.

        The buffers are packed into one floating-point and one integer tensor, so each synchronization uses two
        all-reduces instead of one per buffer.
        """
        if not self.is_multi_gpu:
            return
        buffers = [
            buffer
            for layer in self._actor_batch_norms + self._critic_batch_norms + self._target_batch_norms
            for buffer in layer.buffers()
        ]
        floating = [buffer for buffer in buffers if buffer.is_floating_point()]
        integer = [buffer for buffer in buffers if not buffer.is_floating_point()]
        for group, op in ((floating, dist.ReduceOp.SUM), (integer, dist.ReduceOp.MAX)):
            if not group:
                continue
            packed = torch.cat([buffer.reshape(-1) for buffer in group])
            dist.all_reduce(packed, op=op)
            if op == dist.ReduceOp.SUM:
                packed /= self.gpu_world_size
            for buffer, value in zip(group, packed.split([buffer.numel() for buffer in group])):
                buffer.copy_(value.view_as(buffer))

    def _sample_batch(self) -> _Batch:
        """Build n-step returns, bootstrapping from the next state of the last valid transition."""
        # Accumulate discounted rewards over valid replay transitions
        replay_batch = self.storage.sample(self.batch_size, length=self.num_return_steps)
        rewards = (replay_batch.rewards * self._discounts[: self.num_return_steps] * replay_batch.valid).sum(1)
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

    def release_cuda_graphs(self) -> None:
        """Release the captured CUDA graphs; they are captured again when needed.

        With multiple GPUs, the graphs contain NCCL collectives and must be released before the process group is
        destroyed, which otherwise hangs. :meth:`~rsl_rl.runners.OnPolicyRunner.learn` does this when it returns.
        """
        self._graph_updates.clear()
        self._collection_graphs.clear()
        if self.use_cuda_graph:
            torch.cuda.synchronize(self.device)

    def _run_collection_step(self, key: tuple, fn: Callable[..., dict[str, torch.Tensor]], *args: Any) -> dict:
        """Run a fixed-shape collection function eagerly or, with CUDA graphs enabled, as a replayed graph."""
        if not self.use_cuda_graph:
            return fn(*args)
        if key not in self._collection_graphs:
            self._collection_graphs[key] = CudaGraphCallable(fn, self.device)
        return self._collection_graphs[key](*args)

    def _act(self, obs: TensorDict, uniform: bool) -> dict[str, torch.Tensor]:
        """Sample uniform or exploration actions using only device operations."""
        # Keep policy statistics available to the runner during uniform-action warm-up
        with _BatchNormMode(self._actor_batch_norms, training=False):
            actions = self.actor(obs, stochastic_output=True)

        # Sample uniform actions during warm-up
        distribution = self.actor.distribution
        if uniform:
            uniform_actions = torch.rand(obs.batch_size[0], distribution.output_dim, device=self.device) * 2.0 - 1.0
            actions = distribution.range_offset + distribution.range_scale * uniform_actions  # type: ignore
        elif self.exploration_type == "held":
            # Resample noise and repetition counts for environments whose held noise has expired
            resample = self._noise_steps_left <= 0
            self._noise.copy_(torch.where(resample[:, None], torch.randn_like(self._noise), self._noise))
            repeats = torch.searchsorted(self._noise_repeat_cdf, torch.rand_like(self._noise_steps_left.float()))
            self._noise_steps_left.copy_(torch.where(resample, repeats + 1, self._noise_steps_left) - 1)
            actions = distribution.sample(noise=self._noise)  # type: ignore
        return {"actions": actions}

    def _process_env_step(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        next_observations: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        final_observations: TensorDict,
    ) -> dict[str, torch.Tensor]:
        """Store a transition and update the return statistics and exploration state using only device operations."""
        # Replace reset observations with final observations for completed episodes
        next_observations = TensorDict(
            {
                key: torch.where(dones.reshape(-1, *([1] * (value.ndim - 1))), final_observations[key], value)
                for key, value in next_observations.items()
            },
            batch_size=next_observations.batch_size,
        )

        # Record final observations of completed episodes that are not finite; update() rejects them
        nonfinite = []
        for key in self._obs_groups:
            value = final_observations[key]
            mask = dones.reshape(-1, *([1] * (value.ndim - 1)))
            nonfinite.append(~(torch.isfinite(value) | ~mask).all())
        self._nonfinite_final_obs.logical_or_(torch.stack(nonfinite))

        # Record the transition
        transition = ReplayBuffer.Transition()
        transition.observations = observations
        transition.actions = actions
        transition.rewards = rewards
        transition.next_observations = next_observations
        transition.dones = dones
        transition.terminated = dones & ~time_outs
        self.storage.write_transition(transition)

        # Record discounted returns for reward normalization
        if self.reward_normalization:
            self._returns.mul_(self.gamma).add_(rewards)
            self._max_abs_return.copy_(torch.maximum(self._max_abs_return, self._returns.abs().max()))
            self._return_normalizer.update(self._returns.unsqueeze(-1))
            self._returns.mul_(~dones)

        # Reset exploration for completed episodes
        self._noise_steps_left.mul_(~dones)
        return {}

    def _run_updates(self, phase: int) -> dict[str, torch.Tensor]:
        """Run the gradient updates of one iteration and return the summed losses.

        All operations, including replay sampling and the learning rate schedule, run on the device, so the whole
        method can be captured in a CUDA graph. ``phase`` is the position in the actor update interval of the first
        update.
        """
        totals: dict[str, torch.Tensor] = {}
        for i in range(self.num_updates):
            # Set the scheduled learning rates
            self._set_learning_rates(i)

            # Sample and augment transitions and normalize rewards if enabled
            batch = self._sample_batch()
            if self.reward_normalization:
                batch.rewards = batch.rewards / torch.maximum(
                    self._return_normalizer._std.squeeze() + 1e-8,  # type: ignore
                    self._max_abs_return / self.reward_normalization_max_return,
                )

            # Run the gradient update with the configured actor update interval and sum the losses
            stats = self._gradient_step(batch, update_actor=(phase + i) % self.actor_update_interval == 0)
            for key, value in stats.items():
                totals[key] = totals[key] + value if key in totals else value
        if self.use_cuda_graph:
            self._num_critic_updates_tensor.add_(self.num_updates)
        return totals

    def _learning_rate_factor(self, num_updates: int | torch.Tensor) -> float | torch.Tensor:
        """Return the scheduled learning rate as a fraction of the initial learning rate."""
        if self.learning_rate_schedule == "constant":
            return 1.0
        if isinstance(num_updates, torch.Tensor):
            progress = (num_updates / self.num_schedule_updates).clamp(max=1.0)  # type: ignore
            cosine = torch.cos(math.pi * progress)
        else:
            cosine = math.cos(math.pi * min(num_updates / self.num_schedule_updates, 1.0))  # type: ignore
        return self.final_learning_rate_ratio + (1.0 - self.final_learning_rate_ratio) * 0.5 * (1.0 + cosine)

    def _set_learning_rates(self, offset: int) -> None:
        """Set the learning rates of the update ``offset`` steps after the start of the current iteration."""
        if self.learning_rate_schedule == "constant":
            return
        if self.use_cuda_graph:
            factor = self._learning_rate_factor(self._num_critic_updates_tensor + offset)
        else:
            factor = self._learning_rate_factor(self.num_critic_updates + offset)
        for optimizer, base_learning_rate in zip(
            (self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer), self._base_learning_rates
        ):
            for param_group in optimizer.param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].copy_(base_learning_rate * factor)
                else:
                    param_group["lr"] = base_learning_rate * factor

    def _gradient_step(self, batch: _Batch, update_actor: bool) -> dict[str, torch.Tensor]:
        """Perform one fixed-shape optimizer step."""
        # Replace successors that are not finite with the current observations where they are not bootstrapped
        finite_successors = torch.stack([
            torch.isfinite(value).flatten(1).all(dim=1) for value in batch.next_observations.values()
        ]).all(dim=0)
        use_successors = (batch.bootstrap_discounts != 0) | finite_successors
        batch.next_observations = TensorDict(
            {
                key: torch.where(use_successors.reshape(-1, *([1] * (value.ndim - 1))), value, batch.observations[key])
                for key, value in batch.next_observations.items()
            },
            batch_size=batch.next_observations.batch_size,
        )

        # Update the actor and temperature before the critics
        stats = self._update_actor_and_alpha(batch) if update_actor else {}
        stats.update(self._update_critics(batch))
        self._update_target_critics()
        self._synchronize_batch_norm_buffers()
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
            normalized_actions = self._normalize_actions(actions)
            self.critics.requires_grad_(False)
            try:
                with _BatchNormMode(self._critic_batch_norms, training=False):
                    q_values = torch.stack([
                        self._expected_value(critic(batch.observations, normalized_actions)) for critic in self.critics
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
        self.reduce_parameters(self.actor_optimizer)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        _normalize_parameters(self.actor)

        # Update the entropy temperature
        entropy = -log_probs.detach().float().mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.reduce_parameters(self.alpha_optimizer)
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
            normalized_actions = self._normalize_actions(batch.actions)
            with torch.no_grad():
                with _BatchNormMode(self._actor_batch_norms, training=False):
                    next_actions, next_log_probs = self._sample_actions(batch.next_observations)
                normalized_next_actions = self._normalize_actions(next_actions)
                if len(self._critic_batch_norms) > 0:
                    # Update target batch statistics using current and next observations
                    target_obs = torch.cat([batch.observations, batch.next_observations])
                    target_actions = torch.cat([normalized_actions, normalized_next_actions])
                    with _BatchNormMode(self._target_batch_norms, training=True):
                        next_outputs = [
                            critic(target_obs, target_actions)[batch.actions.shape[0] :]
                            for critic in self.target_critics
                        ]
                else:
                    next_outputs = [
                        critic(batch.next_observations, normalized_next_actions) for critic in self.target_critics
                    ]
                target = self._compute_target(next_outputs, batch, self.log_alpha.detach().exp() * next_log_probs)

            # Evaluate the current state-action values
            if len(self._critic_batch_norms) > 0:
                with _BatchNormMode(self._critic_batch_norms, training=True):
                    outputs = [critic(target_obs, target_actions)[: batch.actions.shape[0]] for critic in self.critics]
            else:
                outputs = [critic(batch.observations, normalized_actions) for critic in self.critics]

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
        self.reduce_parameters(self.critic_optimizer)
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

    def _get_learning_rate_stats(self) -> dict[str, float]:
        """Return the scheduled learning rates of the next update for logging."""
        factor = self._learning_rate_factor(self.num_critic_updates)
        return {
            f"Optimizer/{name}_learning_rate": base_learning_rate * factor
            for name, base_learning_rate in zip(("actor", "critic", "alpha"), self._base_learning_rates)
        }

    def _get_model_stats(self) -> dict[str, torch.Tensor]:
        """Return the entropy temperature and policy standard deviation as device scalars for logging."""
        return {
            "Model/alpha": self.log_alpha.detach().exp().double().reshape(()),
            "Model/mean_policy_std": self.actor.output_std.detach().mean().double(),
        }

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

        # Discard graphs that reference the previous optimizer state; one graph is captured per actor update phase
        self._graph_updates: dict[int, CudaGraphCallable] = {}


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


def _to_floats(values: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert device scalars to floats with a single device synchronization."""
    return dict(zip(values, torch.stack(list(values.values())).tolist()))
