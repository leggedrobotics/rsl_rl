# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
from itertools import chain
from typing import cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class AMPPPO(PPO):
    """PPO with Adversarial Motion Priors (AMP) style discriminator rewards.

    This implementation follows the existing PPO contracts and adds:
    - a discriminator trained to classify expert vs. policy observations,
    - an AMP intrinsic reward computed from discriminator predictions.
    """

    discriminator: MLPModel
    """The AMP discriminator model."""

    amp_rewards: torch.Tensor | None = None
    """Most recent AMP rewards (for debugging/logging)."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        discriminator: MLPModel,
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
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        # AMP parameters
        expert_observations: torch.Tensor | dict[str, torch.Tensor] | TensorDict | None = None,
        expert_observations_path: str | None = None,
        expert_observations_key: str = "expert_observations",
        expert_batch_size: int = 256,
        amp_reward_coef: float = 0.5,
        discriminator_loss_coef: float = 1.0,
        discriminator_learning_rate: float = 1e-3,
        discriminator_optimizer: str = "adam",
        discriminator_max_grad_norm: float = 1.0,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize PPO and AMP discriminator components."""
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.discriminator = discriminator.to(self.device)
        discriminator_optimizer_class = resolve_optimizer(discriminator_optimizer)
        self.discriminator_optimizer = discriminator_optimizer_class(  # type: ignore[operator]
            self.discriminator.parameters(),
            lr=discriminator_learning_rate,
        )

        self.expert_batch_size = expert_batch_size
        self.amp_reward_coef = amp_reward_coef
        self.discriminator_loss_coef = discriminator_loss_coef
        self.discriminator_max_grad_norm = discriminator_max_grad_norm
        self.expert_observations = self._resolve_expert_observations(
            expert_observations=expert_observations,
            expert_observations_path=expert_observations_path,
            expert_observations_key=expert_observations_key,
        )

    def _resolve_expert_observations(
        self,
        expert_observations: torch.Tensor | dict[str, torch.Tensor] | TensorDict | None,
        expert_observations_path: str | None,
        expert_observations_key: str,
    ) -> dict[str, torch.Tensor]:
        """Resolve and validate expert observations from in-memory values or checkpoint files."""
        if expert_observations is None and expert_observations_path is None:
            raise ValueError("AMPPPO requires expert observations via 'expert_observations' or path.")

        resolved: torch.Tensor | dict[str, torch.Tensor] | TensorDict
        if expert_observations is not None:
            resolved = expert_observations
        else:
            if expert_observations_path is None:
                raise ValueError("expert_observations_path cannot be None when expert_observations is not provided.")
            if not os.path.exists(expert_observations_path):
                raise FileNotFoundError(f"Expert observations file not found: {expert_observations_path}")
            loaded = torch.load(expert_observations_path, map_location=self.device, weights_only=False)
            if isinstance(loaded, (torch.Tensor, TensorDict)):
                resolved = loaded
            elif isinstance(loaded, dict):
                if expert_observations_key in loaded:
                    resolved = loaded[expert_observations_key]
                else:
                    resolved = loaded
            else:
                raise ValueError(
                    "Unsupported expert observations file format. Expected tensor, TensorDict, or dict payload."
                )

        if isinstance(resolved, torch.Tensor):
            if len(self.discriminator.obs_groups) != 1:
                raise ValueError(
                    "Tensor expert_observations only supports a single discriminator observation group. "
                    f"Got groups: {self.discriminator.obs_groups}"
                )
            resolved_dict = {self.discriminator.obs_groups[0]: resolved}
        elif isinstance(resolved, TensorDict):
            resolved_dict = {key: resolved[key] for key in self.discriminator.obs_groups}
        else:
            resolved_dict = {key: resolved[key] for key in self.discriminator.obs_groups}

        first_key = self.discriminator.obs_groups[0]
        if first_key not in resolved_dict:
            raise ValueError(f"Missing required expert observation group '{first_key}'.")
        num_samples = resolved_dict[first_key].shape[0]
        if num_samples <= 0:
            raise ValueError("Expert observation dataset is empty.")

        expert_obs: dict[str, torch.Tensor] = {}
        for key in self.discriminator.obs_groups:
            if key not in resolved_dict:
                raise ValueError(
                    f"Missing required expert observation group '{key}'. Available keys: {list(resolved_dict.keys())}"
                )
            tensor = resolved_dict[key].to(self.device)
            if tensor.shape[0] != num_samples:
                raise ValueError(
                    f"All expert observation groups must have equal batch size. '{key}' has {tensor.shape[0]}, "
                    f"expected {num_samples}."
                )
            expert_obs[key] = tensor
        return expert_obs

    def _sample_expert_batch(self, batch_size: int) -> TensorDict:
        """Sample a batch of expert observations for discriminator training."""
        num_samples = next(iter(self.expert_observations.values())).shape[0]
        indices = torch.randint(0, num_samples, (batch_size,), device=self.device)
        data = {key: value[indices] for key, value in self.expert_observations.items()}
        return TensorDict(data, batch_size=[batch_size], device=self.device)

    def _compute_amp_reward(self, obs: TensorDict) -> torch.Tensor:
        """Compute AMP reward from discriminator output.

        The reward is -log(1 - D(s) + eps), where D is the probability that observations are expert-like.
        """
        logits = self.discriminator(obs).squeeze(-1)
        d_prob = torch.sigmoid(logits)
        amp_reward = -torch.log(torch.clamp(torch.ones_like(d_prob) - d_prob, min=1e-6))
        return amp_reward

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and add AMP discriminator reward."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.discriminator.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        with torch.no_grad():
            self.amp_rewards = self._compute_amp_reward(obs)
        self.transition.rewards += self.amp_reward_coef * self.amp_rewards

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def update(self) -> dict[str, float]:
        """Run PPO updates and train the AMP discriminator on expert/policy observations."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_discriminator_loss = 0.0
        mean_amp_reward = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        bce_loss = nn.BCEWithLogitsLoss()
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations,
                        actions=None,
                        env=self.symmetry["_env"],
                    )

                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(obs=None, actions=action_mean_orig, env=self.symmetry["_env"])

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            policy_obs = batch.observations
            assert policy_obs is not None
            policy_obs = cast(TensorDict, policy_obs[:original_batch_size])
            with torch.no_grad():
                current_amp_reward = self._compute_amp_reward(policy_obs)
            mean_amp_reward += current_amp_reward.mean().item()

            expert_obs = self._sample_expert_batch(min(self.expert_batch_size, original_batch_size))
            policy_logits = self.discriminator(policy_obs.detach())
            expert_logits = self.discriminator(expert_obs)

            policy_targets = torch.zeros_like(policy_logits)
            expert_targets = torch.ones_like(expert_logits)
            discriminator_loss = bce_loss(policy_logits, policy_targets) + bce_loss(expert_logits, expert_targets)
            discriminator_loss = self.discriminator_loss_coef * discriminator_loss

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.discriminator_max_grad_norm)
            self.optimizer.step()
            self.discriminator_optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_discriminator_loss += discriminator_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_discriminator_loss /= num_updates
        mean_amp_reward /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp_discriminator": mean_discriminator_loss,
            "amp_reward": mean_amp_reward,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["discriminator_optimizer_state_dict"] = self.discriminator_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load models from a saved dict while supporting PPO-only checkpoints."""
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "discriminator": "discriminator_state_dict" in loaded_dict,
            }

        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd and "rnd_state_dict" in loaded_dict:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if load_cfg.get("discriminator") and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
            if "discriminator_optimizer_state_dict" in loaded_dict:
                self.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> AMPPPO:
        """Construct the AMPPPO algorithm."""
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        discriminator_cfg = cfg.get("discriminator")
        if discriminator_cfg is None:
            discriminator_cfg = {
                "class_name": "MLPModel",
                "hidden_dims": [256, 256],
                "activation": "elu",
            }
            cfg["discriminator"] = discriminator_cfg
        discriminator_class: type[MLPModel] = resolve_callable(discriminator_cfg.pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic", "amp"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        discriminator: MLPModel = discriminator_class(
            obs,
            cfg["obs_groups"],
            "amp",
            1,
            **cfg["discriminator"],
        ).to(device)
        print(f"Discriminator Model: {discriminator}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: AMPPPO = alg_class(
            actor,
            critic,
            discriminator,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [
            self.actor.state_dict(),
            self.critic.state_dict(),
            self.discriminator.state_dict(),
        ]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel