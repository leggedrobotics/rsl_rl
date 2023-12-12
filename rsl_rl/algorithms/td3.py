from __future__ import annotations
import torch
from torch import nn, optim
from typing import Dict, Type, Union

from rsl_rl.algorithms.dpg import AbstractDPG
from rsl_rl.env import VecEnv
from rsl_rl.modules.network import Network
from rsl_rl.storage.storage import Dataset


class TD3(AbstractDPG):
    """Twin-Delayed Deep Deterministic Policy Gradients algorithm.

    This is an implementation of the TD3 algorithm by Fujimoto et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    critic_network: Type[nn.Module] = Network

    def __init__(
        self,
        env: VecEnv,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        target_noise_scale: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)

        self._noise_clip = noise_clip
        self._policy_delay = policy_delay
        self._target_noise_scale = target_noise_scale

        self._register_serializable("_noise_clip", "_policy_delay", "_target_noise_scale")

        self.actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.critic_1 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
        self.critic_2 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)

        self.target_actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.target_critic_1 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
        self.target_critic_2 = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self._update_step = 0

        self._register_serializable(
            "actor",
            "critic_1",
            "critic_2",
            "target_actor",
            "target_critic_1",
            "target_critic_2",
            "actor_optimizer",
            "critic_1_optimizer",
            "critic_2_optimizer",
            "_update_step",
        )

        self.critic = self.critic_1
        self.to(self.device)

    def eval_mode(self) -> TD3:
        super().eval_mode()

        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_actor.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        return self

    def to(self, device: str) -> TD3:
        """Transfers agent parameters to device."""
        super().to(device)

        self.actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_actor.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)

        return self

    def train_mode(self) -> TD3:
        super().train_mode()

        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.target_actor.train()
        self.target_critic_1.train()
        self.target_critic_2.train()

        return self

    def _apply_action_noise(self, actions: torch.Tensor, clip=False) -> torch.Tensor:
        noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * self._action_noise_scale)

        if clip:
            noise = noise.clamp(-self._noise_clip, self._noise_clip)

        noisy_actions = self._process_actions(actions + noise)

        return noisy_actions

    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        super().update(dataset)

        if not self.initialized:
            return {}

        total_actor_loss = torch.zeros(self._batch_count)
        total_critic_1_loss = torch.zeros(self._batch_count)
        total_critic_2_loss = torch.zeros(self._batch_count)

        for idx, batch in enumerate(self.storage.batch_generator(self._batch_size, self._batch_count)):
            actor_obs = batch["actor_observations"]
            critic_obs = batch["critic_observations"]
            actions = batch["actions"].reshape(self._batch_size, -1)
            rewards = batch["rewards"]
            actor_next_obs = batch["next_actor_observations"]
            critic_next_obs = batch["next_critic_observations"]
            dones = batch["dones"]

            critic_1_loss, critic_2_loss = self._update_critic(
                critic_obs, actions, rewards, dones, actor_next_obs, critic_next_obs
            )

            if self._update_step % self._policy_delay == 0:
                evaluation = self.critic_1.forward(
                    self._critic_input(critic_obs, self._process_actions(self.actor.forward(actor_obs)))
                )
                actor_loss = -evaluation.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._update_target(self.actor, self.target_actor)
                self._update_target(self.critic_1, self.target_critic_1)
                self._update_target(self.critic_2, self.target_critic_2)

                total_actor_loss[idx] = actor_loss.item()

            self._update_step = self._update_step + 1

            total_critic_1_loss[idx] = critic_1_loss.item()
            total_critic_2_loss[idx] = critic_2_loss.item()

        stats = {
            "actor": total_actor_loss.mean().item(),
            "critic1": total_critic_1_loss.mean().item(),
            "critic2": total_critic_2_loss.mean().item(),
        }

        return stats

    def _update_critic(self, critic_obs, actions, rewards, dones, actor_next_obs, critic_next_obs):
        target_actor_prediction = self._apply_action_noise(self.target_actor.forward(actor_next_obs), clip=True)
        target_critic_1_prediction = self.target_critic_1.forward(
            self._critic_input(critic_next_obs, target_actor_prediction)
        )
        target_critic_2_prediction = self.target_critic_2.forward(
            self._critic_input(critic_next_obs, target_actor_prediction)
        )
        target_critic_prediction = torch.min(target_critic_1_prediction, target_critic_2_prediction)

        target = (rewards + self._discount_factor * (1 - dones) * target_critic_prediction).detach()

        prediction_1 = self.critic_1.forward(self._critic_input(critic_obs, actions))
        critic_1_loss = (prediction_1 - target).pow(2).mean()

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        prediction_2 = self.critic_2.forward(self._critic_input(critic_obs, actions))
        critic_2_loss = (prediction_2 - target).pow(2).mean()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        return critic_1_loss, critic_2_loss
