from __future__ import annotations
import torch
from torch import optim
from typing import Dict, Union

from rsl_rl.algorithms.dpg import AbstractDPG
from rsl_rl.env import VecEnv
from rsl_rl.modules.network import Network
from rsl_rl.storage.storage import Dataset


class DDPG(AbstractDPG):
    """Deep Deterministic Policy Gradients algorithm.

    This is an implementation of the DDPG algorithm by Lillicrap et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(
        self,
        env: VecEnv,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)

        self.actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.critic = Network(self._critic_input_size, 1, **self._critic_network_kwargs)

        self.target_actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.target_critic = Network(self._critic_input_size, 1, **self._critic_network_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._register_serializable(
            "actor", "critic", "target_actor", "target_critic", "actor_optimizer", "critic_optimizer"
        )

        self.to(self.device)

    def eval_mode(self) -> DDPG:
        super().eval_mode()

        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        return self

    def to(self, device: str) -> DDPG:
        """Transfers agent parameters to device."""
        super().to(device)

        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

        return self

    def train_mode(self) -> DDPG:
        super().train_mode()

        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        return self

    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        super().update(dataset)

        if not self.initialized:
            return {}

        total_actor_loss = torch.zeros(self._batch_count)
        total_critic_loss = torch.zeros(self._batch_count)

        for idx, batch in enumerate(self.storage.batch_generator(self._batch_size, self._batch_count)):
            actor_obs = batch["actor_observations"]
            critic_obs = batch["critic_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            actor_next_obs = batch["next_actor_observations"]
            critic_next_obs = batch["next_critic_observations"]
            dones = batch["dones"]

            target_actor_prediction = self._process_actions(self.target_actor.forward(actor_next_obs))
            target_critic_prediction = self.target_critic.forward(
                self._critic_input(critic_next_obs, target_actor_prediction)
            )

            target = rewards + self._discount_factor * (1 - dones) * target_critic_prediction
            prediction = self.critic.forward(self._critic_input(critic_obs, actions))
            critic_loss = (prediction - target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            evaluation = self.critic.forward(
                self._critic_input(critic_obs, self._process_actions(self.actor.forward(actor_obs)))
            )
            actor_loss = -evaluation.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target(self.actor, self.target_actor)
            self._update_target(self.critic, self.target_critic)

            total_actor_loss[idx] = actor_loss.item()
            total_critic_loss[idx] = critic_loss.item()

        stats = {"actor": total_actor_loss.mean().item(), "critic": total_critic_loss.mean().item()}

        return stats
