import torch
from torch import nn, optim
from typing import Any, Dict, Tuple, Type, Union

from rsl_rl.algorithms.actor_critic import AbstractActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.utils.benchmarkable import Benchmarkable
from rsl_rl.utils.recurrency import trajectories_to_transitions, transitions_to_trajectories
from rsl_rl.modules import GaussianNetwork, Network
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.storage.storage import Dataset


class PPO(AbstractActorCritic):
    """Proximal Policy Optimization algorithm.

    This is an implementation of the PPO algorithm by Schulman et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/1707.06347.pdf

    The implementation works with recurrent neural networks. We implement adaptive learning rate based on the
    KL-divergence between the old and new policy, as described by Schulman et. al. in
    https://arxiv.org/pdf/1707.06347.pdf.
    """

    critic_network: Type[nn.Module] = Network
    _alg_features = dict(recurrent=True)

    schedule_adaptive = "adaptive"
    schedule_fixed = "fixed"

    def __init__(
        self,
        env: VecEnv,
        actor_noise_std: float = 1.0,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.0,
        gae_lambda: float = 0.97,
        gradient_clip: float = 1.0,
        learning_rate: float = 1e-3,
        schedule: str = "fixed",
        target_kl: float = 0.01,
        value_coeff: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            env (VecEnv): A vectorized environment.
            actor_noise_std (float): The standard deviation of the Gaussian noise to add to the actor network output.
            clip_ratio (float): The clipping ratio for the PPO objective.
            entropy_coeff (float): The coefficient for the entropy term in the PPO objective.
            gae_lambda (float): The lambda parameter for the GAE computation.
            gradient_clip (float): The gradient clipping value.
            learning_rate (float): The learning rate for the actor and critic networks.
            schedule (str): The learning rate schedule. Can be "fixed" or "adaptive". Defaults to "fixed".
            target_kl (float): The target KL-divergence for the adaptive learning rate schedule.
            value_coeff (float): The coefficient for the value function loss in the PPO objective.
        """
        kwargs["batch_size"] = env.num_envs
        kwargs["return_steps"] = 1

        super().__init__(env, **kwargs)
        self._critic_input_size = self._critic_input_size - self._action_size  # We use a state-value function (not Q)

        self.storage = RolloutStorage(self.env.num_envs, device=self.device)

        self._register_serializable("storage")

        self._clip_ratio = clip_ratio
        self._entropy_coeff = entropy_coeff
        self._gae_lambda = gae_lambda
        self._gradient_clip = gradient_clip
        self._schedule = schedule
        self._target_kl = target_kl
        self._value_coeff = value_coeff

        self._register_serializable(
            "_clip_ratio",
            "_entropy_coeff",
            "_gae_lambda",
            "_gradient_clip",
            "_schedule",
            "_target_kl",
            "_value_coeff",
        )

        self.actor = GaussianNetwork(
            self._actor_input_size, self._action_size, std_init=actor_noise_std, **self._actor_network_kwargs
        )
        self.critic = self.critic_network(self._critic_input_size, 1, **self._critic_network_kwargs)

        if self.recurrent:
            self.actor.reset_full_hidden_state(batch_size=self.env.num_envs)
            self.critic.reset_full_hidden_state(batch_size=self.env.num_envs)

            tp = lambda v: v.transpose(0, 1)  # for storing, transpose (num_layers, batch, F) to (batch, num_layers, F)
            self.storage.register_processor("actor_state_h", tp)
            self.storage.register_processor("actor_state_c", tp)
            self.storage.register_processor("critic_state_h", tp)
            self.storage.register_processor("critic_state_c", tp)
            self.storage.register_processor("critic_next_state_h", tp)
            self.storage.register_processor("critic_next_state_c", tp)

        self._bm_fuse(self.actor, prefix="actor.")
        self._bm_fuse(self.critic, prefix="critic.")

        self._register_serializable("actor", "critic")

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self._register_serializable("learning_rate", "optimizer")

    def draw_random_actions(self, obs: torch.Tensor, env_info: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError("PPO does not support drawing random actions.")

    @Benchmarkable.register
    def draw_actions(
        self, obs: torch.Tensor, env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        actor_obs, critic_obs = self._process_observations(obs, env_info)

        data = {}

        if self.recurrent:
            data["actor_state_h"] = self.actor.hidden_state[0].detach()
            data["actor_state_c"] = self.actor.hidden_state[1].detach()

        mean, std = self.actor.forward(actor_obs, compute_std=True)
        action_distribution = torch.distributions.Normal(mean, std)
        actions = self._process_actions(action_distribution.rsample()).detach()
        action_prediction_logp = action_distribution.log_prob(actions).sum(-1)

        data["actor_observations"] = actor_obs
        data["critic_observations"] = critic_obs
        data["actions_logp"] = action_prediction_logp.detach()
        data["actions_mean"] = action_distribution.mean.detach()
        data["actions_std"] = action_distribution.stddev.detach()

        return actions, data

    def eval_mode(self) -> AbstractActorCritic:
        super().eval_mode()

        self.actor.eval()
        self.critic.eval()

        return self

    @property
    def initialized(self) -> bool:
        return True

    @Benchmarkable.register
    def process_transition(self, *args) -> Dict[str, torch.Tensor]:
        transition = super().process_transition(*args)

        if self.recurrent:
            transition["critic_state_h"] = self.critic.hidden_state[0].detach()
            transition["critic_state_c"] = self.critic.hidden_state[1].detach()

        transition["values"] = self.critic.forward(transition["critic_observations"]).detach()

        if self.recurrent:
            transition["critic_next_state_h"] = self.critic.hidden_state[0].detach()
            transition["critic_next_state_c"] = self.critic.hidden_state[1].detach()

        return transition

    def parameters(self):
        params = list(self.actor.parameters()) + list(self.critic.parameters())

        return params

    def register_terminations(self, terminations: torch.Tensor) -> None:
        """Registers terminations with the agent.

        Args:
            terminations (torch.Tensor): A 1-dimensional int tensor containing the indices of the terminated
                environments.
        """
        if terminations.shape[0] == 0:
            return

        if self.recurrent:
            self.actor.reset_hidden_state(terminations)
            self.critic.reset_hidden_state(terminations)

    def to(self, device: str) -> AbstractActorCritic:
        super().to(device)

        self.actor.to(device)
        self.critic.to(device)

        return self

    def train_mode(self) -> AbstractActorCritic:
        super().train_mode()

        self.actor.train()
        self.critic.train()

        return self

    @Benchmarkable.register
    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        super().update(dataset)

        assert self.storage.initialized

        total_loss = torch.zeros(self._batch_count)
        total_surrogate_loss = torch.zeros(self._batch_count)
        total_value_loss = torch.zeros(self._batch_count)

        for idx, batch in enumerate(self.storage.batch_generator(self._batch_count, trajectories=self.recurrent)):
            if self.recurrent:
                transition_obs = batch["actor_observations"].reshape(*batch["actor_observations"].shape[:2], -1)
                observations, data = transitions_to_trajectories(transition_obs, batch["dones"])
                hidden_state_h, _ = transitions_to_trajectories(batch["actor_state_h"], batch["dones"])
                hidden_state_c, _ = transitions_to_trajectories(batch["actor_state_c"], batch["dones"])
                # Init. sequence with each trajectory's first hidden state. Subsequent hidden states are produced by the
                # network, depending on the previous hidden state and the current observation.
                hidden_state = (hidden_state_h[0].transpose(0, 1), hidden_state_c[0].transpose(0, 1))

                action_mean, action_std = self.actor.forward(observations, hidden_state=hidden_state, compute_std=True)

                action_mean = action_mean.reshape(*observations.shape[:-1], self._action_size)
                action_std = action_std.reshape(*observations.shape[:-1], self._action_size)

                action_mean = trajectories_to_transitions(action_mean, data)
                action_std = trajectories_to_transitions(action_std, data)
            else:
                action_mean, action_std = self.actor.forward(batch["actor_observations"], compute_std=True)

            actions_dist = torch.distributions.Normal(action_mean, action_std)

            if self._schedule == self.schedule_adaptive:
                self._update_learning_rate(batch, actions_dist)

            surrogate_loss = self._compute_actor_loss(batch, actions_dist)
            value_loss = self._compute_value_loss(batch)
            actions_entropy = actions_dist.entropy().sum(-1)

            loss = surrogate_loss + self._value_coeff * value_loss - self._entropy_coeff * actions_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self._gradient_clip)
            self.optimizer.step()

            total_loss[idx] = loss.detach()
            total_surrogate_loss[idx] = surrogate_loss.detach()
            total_value_loss[idx] = value_loss.detach()

        stats = {
            "total": total_loss.mean().item(),
            "surrogate": total_surrogate_loss.mean().item(),
            "value": total_value_loss.mean().item(),
        }

        return stats

    @Benchmarkable.register
    def _compute_actor_loss(
        self, batch: Dict[str, torch.Tensor], actions_dist: torch.distributions.Normal
    ) -> torch.Tensor:
        batch_actions_logp = actions_dist.log_prob(batch["actions"]).sum(-1)

        ratio = (batch_actions_logp - batch["actions_logp"]).exp()
        surrogate = batch["normalized_advantages"] * ratio
        surrogate_clipped = batch["normalized_advantages"] * ratio.clamp(1.0 - self._clip_ratio, 1.0 + self._clip_ratio)
        surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

        return surrogate_loss

    @Benchmarkable.register
    def _compute_value_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.recurrent:
            observations, data = transitions_to_trajectories(batch["critic_observations"], batch["dones"])
            hidden_state_h, _ = transitions_to_trajectories(batch["critic_state_h"], batch["dones"])
            hidden_state_c, _ = transitions_to_trajectories(batch["critic_state_c"], batch["dones"])
            hidden_states = (hidden_state_h[0].transpose(0, 1), hidden_state_c[0].transpose(0, 1))

            trajectory_evaluations = self.critic.forward(observations, hidden_state=hidden_states)
            trajectory_evaluations = trajectory_evaluations.reshape(*observations.shape[:-1])

            evaluation = trajectories_to_transitions(trajectory_evaluations, data)
        else:
            evaluation = self.critic.forward(batch["critic_observations"])

        value_clipped = batch["values"] + (evaluation - batch["values"]).clamp(-self._clip_ratio, self._clip_ratio)
        returns = batch["advantages"] + batch["values"]
        value_losses = (evaluation - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)

        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        return value_loss

    def _critic_input(self, observations, actions=None) -> torch.Tensor:
        return observations

    def _entry_to_hs(
        self, entry: Dict[str, torch.Tensor], critic: bool = False, next: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper function to turn a dataset entry into a hidden state tuple.

        Args:
            entry (Dict[str, torch.Tensor]): The dataset entry.
            critic (bool): Whether to extract the hidden state for the critic instead of the actor. Defaults to False.
            next (bool): Whether the hidden state is for the next step or the current. Defaults to False.
        Returns:
            A tuple of hidden state tensors.
        """
        key = ("critic" if critic else "actor") + "_" + ("next_state" if next else "state")
        hidden_state = entry[f"{key}_h"], entry[f"{key}_c"]

        return hidden_state

    @Benchmarkable.register
    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Processes a dataset before it is added to the replay memory.

        Computes advantages and returns.

        Args:
            dataset (Dataset): The dataset to process.
        Returns:
            A Dataset object containing the processed data.
        """
        rewards = torch.stack([entry["rewards"] for entry in dataset])
        dones = torch.stack([entry["dones"] for entry in dataset])
        timeouts = torch.stack([entry["timeouts"] for entry in dataset])
        values = torch.stack([entry["values"] for entry in dataset])

        # We could alternatively compute the next hidden state from the current state and hidden state. But this
        # (storing the hidden state when evaluating the action in process_transition) is computationally more efficient
        # and doesn't change the result as the network is not updated between storing the data and computing advantages.

        critic_kwargs = (
            {"hidden_state": (dataset[-1]["critic_state_h"], dataset[-1]["critic_state_c"])} if self.recurrent else {}
        )
        final_values = self.critic.forward(dataset[-1]["next_critic_observations"], **critic_kwargs)
        next_values = torch.cat((values[1:], final_values.unsqueeze(0)), dim=0)

        rewards += self.gamma * timeouts * values
        deltas = (rewards + (1 - dones).float() * self.gamma * next_values - values).reshape(-1, self.env.num_envs)

        advantages = torch.zeros((len(dataset) + 1, self.env.num_envs), device=self.device)
        for step in reversed(range(len(dataset))):
            advantages[step] = (
                deltas[step] + (1 - dones[step]).float() * self.gamma * self._gae_lambda * advantages[step + 1]
            )
        advantages = advantages[:-1]

        amean, astd = advantages.mean(), torch.nan_to_num(advantages.std())
        for step in range(len(dataset)):
            dataset[step]["advantages"] = advantages[step]
            dataset[step]["normalized_advantages"] = (advantages[step] - amean) / (astd + 1e-8)

        return dataset

    @Benchmarkable.register
    def _update_learning_rate(self, batch: Dict[str, torch.Tensor], actions_dist: torch.distributions.Normal) -> None:
        with torch.inference_mode():
            actions_mean = actions_dist.mean
            actions_std = actions_dist.stddev

            kl = torch.sum(
                torch.log(actions_std / batch["actions_std"] + 1.0e-5)
                + (torch.square(batch["actions_std"]) + torch.square(batch["actions_mean"] - actions_mean))
                / (2.0 * torch.square(actions_std))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self._target_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self._target_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
