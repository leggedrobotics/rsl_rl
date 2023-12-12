from __future__ import annotations
import torch
from torch import nn
from typing import Type

from rsl_rl.algorithms.td3 import TD3
from rsl_rl.env import VecEnv
from rsl_rl.modules import QuantileNetwork


class DTD3(TD3):
    """Distributional Twin-Delayed Deep Deterministic Policy Gradients algorithm.

    This is an implementation of the TD3 algorithm by Fujimoto et. al. for vectorized environments using a QR-DQN
    critic.
    """

    critic_network: Type[nn.Module] = QuantileNetwork

    def __init__(
        self,
        env: VecEnv,
        quantile_count: int = 200,
        **kwargs,
    ) -> None:
        self._quantile_count = quantile_count
        self._register_critic_network_kwargs(quantile_count=self._quantile_count)

        super().__init__(env, **kwargs)

    def _update_critic(self, critic_obs, actions, rewards, dones, actor_next_obs, critic_next_obs):
        target_action = self._apply_action_noise(self.target_actor.forward(actor_next_obs), clip=True)
        target_critic_input = self._critic_input(critic_next_obs, target_action)
        target_critic_prediction_1 = self.target_critic_1.forward(target_critic_input, distribution=True)
        target_critic_prediction_2 = self.target_critic_2.forward(target_critic_input, distribution=True)
        target_critic_prediction = torch.minimum(target_critic_prediction_1, target_critic_prediction_2)

        target = (
            rewards.reshape(-1, 1) + self._discount_factor * (1 - dones).reshape(-1, 1) * target_critic_prediction
        ).detach()

        critic_input = self._critic_input(critic_obs, actions).detach()
        critic_1_prediction = self.critic_1.forward(critic_input, distribution=True)
        critic_1_loss = self.critic_1.quantile_huber_loss(critic_1_prediction, target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_prediction = self.critic_2.forward(critic_input, distribution=True)
        critic_2_loss = self.critic_2.quantile_huber_loss(critic_2_prediction, target)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        return critic_1_loss, critic_2_loss
