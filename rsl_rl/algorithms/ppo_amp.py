from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent, AMPDiscriminator
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import string_to_callable
from rsl_rl.algorithms import PPO
from rsl_rl.modules.amp import LossType


class PPOAMP(PPO):

    policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticCNN,
        storage: RolloutStorage,
        disc_obs_buffer: CircularBuffer, 
        disc_demo_obs_buffer: CircularBuffer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy,
            storage,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            normalize_advantage_per_mini_batch,
            device,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )
        
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for PPOAMP algorithm.")
        
        if self.amp_cfg["loss_type"] == "GAN":
            self.loss_type = LossType.GAN
        elif self.amp_cfg["loss_type"] == "LSGAN":
            self.loss_type = LossType.LSGAN
        elif self.amp_cfg["loss_type"] == "WGAN":
            self.loss_type = LossType.WGAN
        else:
            raise ValueError(f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'GAN', 'LSGAN', or 'WGAN'")
        
        self.amp_discriminator: AMPDiscriminator = AMPDiscriminator(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=self.policy.obs_groups,
            loss_type=self.loss_type,
            device=device,
            **self.amp_cfg.get("amp_discriminator", {})
        ).to(self.device)
        
        # optimizer for policy and discriminator
        params = [
            {
                "name": "disc_trunk", 
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],  # L2 regularization for the discriminator trunk
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],  # L2 regularization for the discriminator linear layer
            }
        ]
        # use a separate optimizer for the AMP discriminator
        self.disc_optimizer = optim.Adam(
            params,
            lr=self.amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)
        
        # Storage for AMP discriminator observations
        self.disc_obs_buffer: CircularBuffer = disc_obs_buffer
        self.disc_demo_obs_buffer: CircularBuffer = disc_demo_obs_buffer
        
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
        # Compute the Style Reward
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc_obs, dt=self.amp_cfg["step_dt"])
        # Linearly interpolate between task reward and style reward
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)
        # Store the un-normalized disc obs and disc demo obs into buffers
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
        # Call the parent class method with the new rewards
        super().process_env_step(obs, self.rewards_lerp, dones, extras)
        
    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # AMP discriminator loss and other info
        mean_disc_loss = 0
        mean_disc_grad_penalty = 0
        mean_disc_score = 0
        mean_disc_demo_score = 0

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env, # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        # Iterate over batches
        for samples, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = samples
            
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
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
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # AMP discriminator loss
            with torch.no_grad():
                disc_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch) # [mini_batch_size, disc_obs_steps, disc_obs_dim]
                disc_demo_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)
            
            mini_batch_size = disc_obs_batch_normed.shape[0]
            disc_score = self.amp_discriminator(disc_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            disc_demo_score = self.amp_discriminator(disc_demo_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            
            if self.loss_type == LossType.GAN:
                bce = torch.nn.BCEWithLogitsLoss()
                policy_loss = bce(
                    disc_score, torch.zeros_like(disc_score, device=self.device)
                )
                demo_loss = bce(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.LSGAN:
                policy_loss = torch.nn.MSELoss()(
                    disc_score, -1 * torch.ones_like(disc_score, device=self.device)
                )
                demo_loss = torch.nn.MSELoss()(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.WGAN:
                disc_loss = - torch.mean(disc_demo_score) + torch.mean(disc_score)
            else: 
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'")

            disc_grad_penalty = self.amp_discriminator.compute_grad_penalty(
                demo_data=disc_demo_obs_batch_normed.reshape(mini_batch_size, -1),
                scale=self.amp_cfg["grad_penalty_scale"]
            )
            disc_total_loss = disc_loss + disc_grad_penalty

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            # Compute the gradients for AMP discriminator
            self.disc_optimizer.zero_grad()
            disc_total_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # Apply the gradients for AMP discriminator
            self.disc_optimizer.step()
            # Update the AMP normalizer
            self.amp_discriminator.update_normalization(disc_obs_batch)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # AMP discriminator loss and other info
            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += disc_grad_penalty.item()
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_disc_grad_penalty /= num_updates
        mean_disc_score /= num_updates
        mean_disc_demo_score /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["amp/disc_loss"] = mean_disc_loss
        loss_dict["amp/disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["amp/disc_score"] = mean_disc_score
        loss_dict["amp/disc_demo_score"] = mean_disc_demo_score

        return loss_dict