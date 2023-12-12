from __future__ import annotations
import torch
from typing import Dict, Union
from rsl_rl.algorithms.dpg import AbstractDPG
from rsl_rl.env import VecEnv
from rsl_rl.storage.storage import Dataset

from rsl_rl.modules import CategoricalNetwork, Network


class D4PG(AbstractDPG):
    """Distributed Distributional Deep Deterministic Policy Gradients algorithm.

    This is an implementation of the D4PG algorithm by Barth-Maron et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/1804.08617.pdf
    """

    def __init__(
        self,
        env: VecEnv,
        actor_lr: float = 1e-4,
        atom_count: int = 51,
        critic_activations: list = ["relu", "relu", "relu"],
        critic_lr: float = 1e-3,
        target_update_delay: int = 2,
        value_max: float = 10.0,
        value_min: float = -10.0,
        **kwargs,
    ) -> None:
        """
        Args:
            env (VecEnv): A vectorized environment.
            actor_lr (float): The learning rate for the actor network.
            atom_count (int): The number of atoms to use for the categorical distribution.
            critic_activations (list): A list of activation functions to use for the critic network.
            critic_lr (float): The learning rate for the critic network.
            target_update_delay (int): The number of steps to wait before updating the target networks.
            value_max (float): The maximum value for the categorical distribution.
            value_min (float): The minimum value for the categorical distribution.
        """
        kwargs["critic_activations"] = critic_activations

        super().__init__(env, **kwargs)

        self._atom_count = atom_count
        self._target_update_delay = target_update_delay
        self._value_max = value_max
        self._value_min = value_min

        self._register_serializable("_atom_count", "_target_update_delay", "_value_max", "_value_min")

        self.actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.critic = CategoricalNetwork(
            self._critic_input_size,
            1,
            atom_count=atom_count,
            value_max=value_max,
            value_min=value_min,
            **self._critic_network_kwargs,
        )

        self.target_actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.target_critic = CategoricalNetwork(
            self._critic_input_size,
            1,
            atom_count=atom_count,
            value_max=value_max,
            value_min=value_min,
            **self._critic_network_kwargs,
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._register_serializable(
            "actor", "critic", "target_actor", "target_critic", "actor_optimizer", "critic_optimizer"
        )

        self._update_step = 0

        self._register_serializable("_update_step")

        self.to(self.device)

    def eval_mode(self) -> D4PG:
        super().eval_mode()

        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        return self

    def to(self, device: str) -> D4PG:
        super().to(device)

        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

        return self

    def train_mode(self) -> D4PG:
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
            actions = batch["actions"].reshape(self._batch_size, -1)
            rewards = batch["rewards"]
            actor_next_obs = batch["next_actor_observations"]
            critic_next_obs = batch["next_critic_observations"]
            dones = batch["dones"]

            predictions = self.critic.forward(self._critic_input(critic_obs, actions), distribution=True).squeeze()
            target_actor_prediction = self._process_actions(self.target_actor.forward(actor_next_obs))
            target_probabilities = self.target_critic.forward(
                self._critic_input(critic_next_obs, target_actor_prediction), distribution=True
            ).squeeze()
            targets = self.target_critic.compute_targets(rewards, dones, self._discount_factor)
            critic_loss = self.target_critic.categorical_loss(predictions, target_probabilities, targets)

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

            if self._update_step % self._target_update_delay == 0:
                self._update_target(self.actor, self.target_actor)
                self._update_target(self.critic, self.target_critic)

            self._update_step += 1

            total_actor_loss[idx] = actor_loss.item()
            total_critic_loss[idx] = critic_loss.item()

        stats = {"actor": total_actor_loss.mean().item(), "critic": total_critic_loss.mean().item()}

        return stats
