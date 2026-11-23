# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    CudaGraphCallable,
    compile_model,
    reduce_gradients_in_buckets,
    resolve_class,
    resolve_obs_groups,
    resolve_optimizer,
)


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        use_cuda_graph: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Validate configuration
        if actor.distribution is None:
            raise ValueError("PPO requires an actor with an output distribution.")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError(f"Unsupported learning rate schedule: {schedule}. Expected 'fixed' or 'adaptive'.")
        if use_cuda_graph and torch.device(device).type != "cuda":
            raise ValueError("CUDA graph capture requires a CUDA device.")
        if use_cuda_graph and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("CUDA graph capture does not yet support recurrent models.")
        if use_cuda_graph and rnd_cfg:
            raise ValueError("CUDA graph capture does not yet support RND.")
        if use_cuda_graph and multi_gpu_cfg is not None:
            raise ValueError("CUDA graph capture does not yet support multi-GPU training.")

        # Device-related parameters
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            list(dict.fromkeys(chain(self.actor.parameters(), self.critic.parameters()))), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_mixed_precision = use_mixed_precision

        self._update_graph: CudaGraphCallable | None = None
        self._configure_optimizer_capture()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Record one environment step and return per-environment values that the logger sums over each episode."""
        step_dict = {"Training/mean_reward": rewards}
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            step_dict["Training/mean_reward"] = rewards + self.intrinsic_rewards
            step_dict["Rnd/mean_extrinsic_reward"] = rewards
            step_dict["Rnd/mean_intrinsic_reward"] = self.intrinsic_rewards
        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )
        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
        return step_dict

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute values for the last step
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        # Restore the critic's hidden state so the next rollout is not affected by the forward pass
        self.critic.reset(hidden_state=critic_hidden_state)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        torch.sub(st.returns, st.values, out=st.advantages)
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch and st.advantages.numel() > 1:
            mean, std = st.advantages.mean(), st.advantages.std()
            st.advantages.sub_(mean).div_(std + 1e-8)

    def update(self) -> tuple[dict[str, float], dict[str, float]]:
        """Run optimization epochs over stored batches and return mean losses and statistics."""
        if self.use_cuda_graph:
            if self._update_graph is None:
                self._update_graph = CudaGraphCallable(self._update_mini_batches, self.device)
            totals = self._update_graph()
        else:
            totals = self._update_mini_batches()

        # Update the normalizers
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)  # type: ignore
        self.critic.update_normalization(obs)  # type: ignore
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

        device_stats = {"Model/mean_policy_std": self.actor.output_std.mean()}
        if self.use_cuda_graph:
            device_stats["Optimizer/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics = {**totals, **device_stats}
        metrics = dict(zip(metrics, torch.stack([value.float() for value in metrics.values()]).tolist()))
        if self.use_cuda_graph:
            self.learning_rate = metrics["Optimizer/learning_rate"]

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {key: metrics[key] / num_updates for key in totals}

        # Construct the stats dictionary
        stats_dict = {
            "Optimizer/learning_rate": self.learning_rate,
            "Model/mean_policy_std": metrics["Model/mean_policy_std"],
        }
        if self.rnd:
            stats_dict["Rnd/rnd_weight"] = self.rnd.weight

        # Clear the storage
        self.storage.clear()

        return loss_dict, stats_dict

    def _update_mini_batches(self) -> dict[str, torch.Tensor]:
        """Run the optimization epochs and return the summed losses as device scalars."""
        totals = {key: torch.zeros((), device=self.device) for key in ("value", "surrogate", "entropy")}
        # RND loss
        if self.rnd:
            totals["rnd"] = torch.zeros((), device=self.device)
        # Symmetry loss
        if self.symmetry:
            totals["symmetry"] = torch.zeros((), device=self.device)

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch and batch.advantages.numel() > 1:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # Recompute actions log prob and entropy for current batch of transitions
                # Note: We need to do this because we updated the policy with new parameters
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
                distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy[:original_batch_size]

                # Compute KL divergence and adapt the learning rate
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                        kl_mean = torch.mean(kl)

                        # Reduce the KL divergence across all GPUs
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        # Update the learning rate only on the main process
                        if self.use_cuda_graph:
                            self._adapt_learning_rate_on_device(kl_mean)
                        elif self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        # Update the learning rate for all GPUs
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        # Update the learning rate for all parameter groups
                        if not self.use_cuda_graph:
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))  # type: ignore
                surrogate = -batch.advantages.squeeze(-1) * ratio  # type: ignore
                surrogate_clipped = -batch.advantages.squeeze(-1) * torch.clamp(  # type: ignore
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # RND loss
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

                # Symmetry loss
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                    if self.symmetry.use_mirror_loss:
                        loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad(set_to_none=not self.use_cuda_graph)
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            totals["value"] += value_loss.detach()
            totals["surrogate"] += surrogate_loss.detach()
            totals["entropy"] += entropy_loss.detach()
            # RND loss
            if self.rnd:
                totals["rnd"] += rnd_loss.detach()  # type: ignore
            # Symmetry loss
            if self.symmetry:
                totals["symmetry"] += symmetry_loss.detach()

        return totals

    def _adapt_learning_rate_on_device(self, kl_mean: torch.Tensor) -> None:
        """Apply the adaptive learning rate rule to the optimizer's device learning rate without synchronizing."""
        learning_rate = self.optimizer.param_groups[0]["lr"]
        increase = (kl_mean < self.desired_kl / 2.0) & (kl_mean > 0.0)
        new_learning_rate = torch.where(
            kl_mean > self.desired_kl * 2.0,
            torch.clamp(learning_rate / 1.5, min=1e-5),
            torch.where(increase, torch.clamp(learning_rate * 1.5, max=1e-2), learning_rate),
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"].copy_(new_learning_rate)

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
            saved_dict["rnd_update_counter"] = self.rnd.update_counter
        return saved_dict

    def get_default_load_cfg(self) -> dict:
        """Return the default selection for resuming this algorithm's own checkpoints."""
        return {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": True}

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None:
            load_cfg = self.get_default_load_cfg()

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self._configure_optimizer_capture()
            self.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            self.rnd.update_counter = loaded_dict.get("rnd_update_counter", 0)
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables and configs
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the actor and critic
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if alg_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        models = [self._raw_actor, self._raw_critic]
        if self.rnd:
            models += [self.rnd.predictor, self.rnd.target]
        for model in models:
            for tensor in model.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            params = chain(params, self.rnd.parameters())
        # Average the gradients across all GPUs in bounded buckets
        reduce_gradients_in_buckets(params, self.gpu_world_size, self.grad_reduce_bucket_mb)

    def _configure_optimizer_capture(self) -> None:
        """Configure the optimizer's learning rate and step counters for CUDA graph capture."""
        for param_group in self.optimizer.param_groups:
            if self.use_cuda_graph and "capturable" not in param_group:
                raise ValueError("CUDA graphs require a capturable optimizer such as Adam or AdamW.")
            if "capturable" in param_group:
                param_group["capturable"] = self.use_cuda_graph
            if self.use_cuda_graph:
                param_group["lr"] = torch.as_tensor(param_group["lr"], device=self.device)
            else:
                param_group["lr"] = float(param_group["lr"])
        for state in self.optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].to(self.device if self.use_cuda_graph else "cpu")
        self._update_graph = None
