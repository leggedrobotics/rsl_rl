# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class DaggerPPO(PPO):
	"""PPO with an auxiliary DAgger imitation loss from a fixed teacher policy."""

	teacher: MLPModel
	"""The teacher policy."""

	teacher_loaded: bool = False
	"""Indicates whether the teacher parameters have been loaded."""

	def __init__(
		self,
		actor: MLPModel,
		critic: MLPModel,
		teacher: MLPModel,
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
		rnd_cfg: dict | None = None,
		symmetry_cfg: dict | None = None,
		dagger_loss_coef: float = 1.0,
		loss_type: str = "mse",
		device: str = "cpu",
		multi_gpu_cfg: dict | None = None,
	) -> None:
		"""Initialize PPO and add a fixed teacher policy for imitation learning."""
		if teacher.is_recurrent:
			raise ValueError("DaggerPPO does not support recurrent teacher policies.")

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

		self.teacher = teacher.to(self.device)
		self.teacher.requires_grad_(False)
		self.dagger_loss_coef = dagger_loss_coef

		loss_fn_dict = {
			"mse": nn.functional.mse_loss,
			"huber": nn.functional.huber_loss,
		}
		if loss_type not in loss_fn_dict:
			raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")
		self.loss_fn = loss_fn_dict[loss_type]

	def update(self) -> dict[str, float]:
		"""Run PPO updates with an additional teacher-imitation loss."""
		mean_value_loss = 0.0
		mean_surrogate_loss = 0.0
		mean_entropy = 0.0
		mean_dagger_loss = 0.0
		mean_rnd_loss = 0.0 if self.rnd else None
		mean_symmetry_loss = 0.0 if self.symmetry else None

		if self.actor.is_recurrent or self.critic.is_recurrent:
			generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
		else:
			generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

		for batch in generator:
			original_batch_size = batch.observations.batch_size[0]

			if self.normalize_advantage_per_mini_batch:
				with torch.no_grad():
					batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

			teacher_observations = batch.observations
			assert teacher_observations is not None

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
						obs=batch.observations, actions=None, env=self.symmetry["_env"]
					)

				mean_actions = self.actor(batch.observations.detach().clone())
				action_mean_orig = mean_actions[:original_batch_size]
				_, actions_mean_symm = data_augmentation_func(
					obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
				)

				mse_loss = torch.nn.MSELoss()
				symmetry_loss = mse_loss(
					mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
				)
				if self.symmetry["use_mirror_loss"]:
					loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
				else:
					symmetry_loss = symmetry_loss.detach()

			actor_observations = teacher_observations[:original_batch_size]
			actor_hidden_state = None if batch.hidden_states[0] is None else batch.hidden_states[0]
			student_actions = self.actor(
				actor_observations,
				masks=batch.masks,
				hidden_state=actor_hidden_state,
			)
			with torch.no_grad():
				teacher_actions = self.teacher(actor_observations, masks=batch.masks)
			dagger_loss = self.loss_fn(student_actions, teacher_actions)
			loss += self.dagger_loss_coef * dagger_loss

			if self.rnd:
				with torch.no_grad():
					rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
					rnd_state = self.rnd.state_normalizer(rnd_state)
				predicted_embedding = self.rnd.predictor(rnd_state)
				target_embedding = self.rnd.target(rnd_state).detach()
				mseloss = torch.nn.MSELoss()
				rnd_loss = mseloss(predicted_embedding, target_embedding)

			self.optimizer.zero_grad()
			loss.backward()
			if self.rnd:
				self.rnd_optimizer.zero_grad()
				rnd_loss.backward()

			if self.is_multi_gpu:
				self.reduce_parameters()

			nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
			nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
			self.optimizer.step()
			if self.rnd_optimizer:
				self.rnd_optimizer.step()

			mean_value_loss += value_loss.item()
			mean_surrogate_loss += surrogate_loss.item()
			mean_entropy += entropy.mean().item()
			mean_dagger_loss += dagger_loss.item()
			if mean_rnd_loss is not None:
				mean_rnd_loss += rnd_loss.item()
			if mean_symmetry_loss is not None:
				mean_symmetry_loss += symmetry_loss.item()

		num_updates = self.num_learning_epochs * self.num_mini_batches
		mean_value_loss /= num_updates
		mean_surrogate_loss /= num_updates
		mean_entropy /= num_updates
		mean_dagger_loss /= num_updates
		if mean_rnd_loss is not None:
			mean_rnd_loss /= num_updates
		if mean_symmetry_loss is not None:
			mean_symmetry_loss /= num_updates

		self.storage.clear()

		loss_dict = {
			"value": mean_value_loss,
			"surrogate": mean_surrogate_loss,
			"entropy": mean_entropy,
			"dagger": mean_dagger_loss,
		}
		if self.rnd:
			loss_dict["rnd"] = mean_rnd_loss
		if self.symmetry:
			loss_dict["symmetry"] = mean_symmetry_loss

		return loss_dict

	def train_mode(self) -> None:
		"""Set train mode for actor and critic while keeping the teacher frozen."""
		super().train_mode()
		self.teacher.eval()

	def eval_mode(self) -> None:
		"""Set evaluation mode for all models."""
		super().eval_mode()
		self.teacher.eval()

	def save(self) -> dict:
		"""Return a dict of all models for saving."""
		saved_dict = super().save()
		saved_dict["teacher_state_dict"] = self.teacher.state_dict()
		return saved_dict

	def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
		"""Load actor-critic state and optionally load teacher from PPO or DaggerPPO checkpoints."""
		if load_cfg is None and "teacher_state_dict" not in loaded_dict and "actor_state_dict" in loaded_dict:
			load_cfg = {"teacher": True, "iteration": False}
		elif load_cfg is None:
			load_cfg = {
				"actor": True,
				"critic": True,
				"optimizer": True,
				"iteration": True,
				"rnd": True,
				"teacher": True,
			}

		if load_cfg.get("actor"):
			self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
		if load_cfg.get("critic"):
			self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
		if load_cfg.get("optimizer"):
			self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
		if load_cfg.get("rnd") and self.rnd:
			self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
			self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
		if load_cfg.get("teacher"):
			teacher_state_dict = loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"]
			self.teacher.load_state_dict(teacher_state_dict, strict=strict)
			self.teacher_loaded = True

		return load_cfg.get("iteration", False)

	@staticmethod
	def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DaggerPPO:
		"""Construct the DAgger-PPO algorithm."""
		alg_class: type[DaggerPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
		actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
		critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
		teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

		default_sets = ["actor", "critic", "teacher"]
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
		teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
			device
		)
		print(f"Teacher Model: {teacher}")

		storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

		alg: DaggerPPO = alg_class(
			actor,
			critic,
			teacher,
			storage,
			device=device,
			**cfg["algorithm"],
			multi_gpu_cfg=cfg["multi_gpu"],
		)

		return alg

	def broadcast_parameters(self) -> None:
		"""Broadcast model parameters to all GPUs."""
		model_params = [self.actor.state_dict(), self.critic.state_dict(), self.teacher.state_dict()]
		if self.rnd:
			model_params.append(self.rnd.predictor.state_dict())
		torch.distributed.broadcast_object_list(model_params, src=0)
		self.actor.load_state_dict(model_params[0])
		self.critic.load_state_dict(model_params[1])
		self.teacher.load_state_dict(model_params[2])
		if self.rnd:
			self.rnd.predictor.load_state_dict(model_params[3])
