import torch
from torch import nn
from typing import Tuple, Type


from rsl_rl.algorithms.sac import SAC
from rsl_rl.env import VecEnv
from rsl_rl.modules.quantile_network import QuantileNetwork


class DSAC(SAC):
    """Deep Soft Actor Critic (DSAC) algorithm.

    This is an implementation of the DSAC algorithm by Ma et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/2004.14547.pdf

    The implementation inherits automatic tuning of the temperature parameter (alpha) and tanh action scaling from
    the SAC implementation.
    """

    critic_network: Type[nn.Module] = QuantileNetwork

    def __init__(self, env: VecEnv, critic_activations=["relu", "relu", "relu"], quantile_count=200, **kwargs):
        """
        Args:
            env (VecEnv): A vectorized environment.
            critic_activations (list): A list of activation functions to use for the critic network.
            quantile_count (int): The number of quantiles to use for the critic QR network.
        """
        self._quantile_count = quantile_count
        self._register_critic_network_kwargs(quantile_count=self._quantile_count)

        kwargs["critic_activations"] = critic_activations

        super().__init__(env, **kwargs)

    def _update_critic(
        self,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        actor_next_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_action, target_action_logp = self._sample_action(actor_next_obs)
        target_critic_input = self._critic_input(critic_next_obs, target_action)
        target_critic_prediction_1 = self.target_critic_1.forward(target_critic_input, distribution=True)
        target_critic_prediction_2 = self.target_critic_2.forward(target_critic_input, distribution=True)
        target_critic_prediction = torch.minimum(target_critic_prediction_1, target_critic_prediction_2)

        next_soft_q = target_critic_prediction - self.alpha * target_action_logp.unsqueeze(-1).repeat(
            1, self._quantile_count
        )
        target = (rewards.reshape(-1, 1) + self._discount_factor * (1 - dones).reshape(-1, 1) * next_soft_q).detach()

        critic_input = self._critic_input(critic_obs, actions).detach()
        critic_1_prediction = self.critic_1.forward(critic_input, distribution=True)
        critic_1_loss = self.critic_1.quantile_huber_loss(critic_1_prediction, target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), self._gradient_clip)
        self.critic_1_optimizer.step()

        critic_2_prediction = self.critic_2.forward(critic_input, distribution=True)
        critic_2_loss = self.critic_2.quantile_huber_loss(critic_2_prediction, target)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), self._gradient_clip)
        self.critic_2_optimizer.step()

        return critic_1_loss, critic_2_loss
