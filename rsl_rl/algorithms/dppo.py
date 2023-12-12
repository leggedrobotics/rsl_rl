import torch
from torch import nn
from typing import Dict, List, Tuple, Type, Union

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.distributions import QuantileDistribution
from rsl_rl.env import VecEnv
from rsl_rl.utils.benchmarkable import Benchmarkable
from rsl_rl.utils.recurrency import trajectories_to_transitions, transitions_to_trajectories
from rsl_rl.modules import ImplicitQuantileNetwork, QuantileNetwork
from rsl_rl.storage.storage import Dataset


class DPPO(PPO):
    """Distributional Proximal Policy Optimization algorithm.

    This algorithm is an extension of PPO that uses a distributional method (either QR-DQN or IQN) to estimate the
    value function.

    QR-DQN Paper: https://arxiv.org/pdf/1710.10044.pdf
    IQN Paper: https://arxiv.org/pdf/1806.06923.pdf

    The implementation works with recurrent neural networks. We further implement Sample-Replacement SR(lambda) for the
    value target computation, as described by Nam et. al. in https://arxiv.org/pdf/2105.11366.pdf.
    """

    critic_network: Type[nn.Module] = QuantileNetwork
    _alg_features = dict(recurrent=True)

    value_loss_energy = "sample_energy"
    value_loss_l1 = "quantile_l1"
    value_loss_huber = "quantile_huber"

    network_qrdqn = "qrdqn"
    network_iqn = "iqn"

    networks = {network_qrdqn: QuantileNetwork, network_iqn: ImplicitQuantileNetwork}

    values_losses = {
        network_qrdqn: {
            value_loss_energy: QuantileNetwork.sample_energy_loss,
            value_loss_l1: QuantileNetwork.quantile_l1_loss,
            value_loss_huber: QuantileNetwork.quantile_huber_loss,
        },
        network_iqn: {
            value_loss_energy: ImplicitQuantileNetwork.sample_energy_loss,
        },
    }

    def __init__(
        self,
        env: VecEnv,
        critic_activations: List[str] = ["relu", "relu", "relu"],
        critic_network: str = network_qrdqn,
        iqn_action_samples: int = 32,
        iqn_embedding_size: int = 64,
        iqn_feature_layers: int = 1,
        iqn_value_samples: int = 8,
        qrdqn_quantile_count: int = 200,
        value_lambda: float = 0.95,
        value_loss: str = value_loss_l1,
        value_loss_kwargs: Dict = {},
        value_measure: str = None,
        value_measure_adaptation: Union[Tuple, None] = None,
        value_measure_kwargs: Dict = {},
        **kwargs,
    ):
        """
        Args:
            env (VecEnv): A vectorized environment.
            critic_activations (List[str]): A list of activations to use for the critic network.
            critic_network (str): The critic network to use.
            iqn_action_samples (int): The number of samples to use for the critic IQN network when acting.
            iqn_embedding_size (int): The embedding size to use for the critic IQN network.
            iqn_feature_layers (int): The number of feature layers to use for the critic IQN network.
            iqn_value_samples (int): The number of samples to use for the critic IQN network when computing the value.
            qrdqn_quantile_count (int): The number of quantiles to use for the critic QR network.
            value_lambda (float): The lambda parameter for the SR(lambda) value target computation.
            value_loss (str): The loss function to use for the critic network.
            value_loss_kwargs (Dict): Keyword arguments for computing the value loss.
            value_measure (str): The probability measure to apply to the critic network output distribution when
                updating the policy.
            value_measure_adaptation (Union[Tuple, None]): Controls adaptation of the value measure. If None, no
                adaptation is performed. If a tuple, the tuple specifies the observations that are passed to the value
                measure.
            value_measure_kwargs (Dict): The keyword arguments to pass to the value measure.
        """
        self._register_critic_network_kwargs(measure=value_measure, measure_kwargs=value_measure_kwargs)

        self._critic_network_name = critic_network
        self.critic_network = self.networks[self._critic_network_name]
        if self._critic_network_name == self.network_qrdqn:
            self._register_critic_network_kwargs(quantile_count=qrdqn_quantile_count)
        elif self._critic_network_name == self.network_iqn:
            self._register_critic_network_kwargs(feature_layers=iqn_feature_layers, embedding_size=iqn_embedding_size)

        kwargs["critic_activations"] = critic_activations

        if value_measure_adaptation is not None:
            # Value measure adaptation observations are not passed to the critic network.
            kwargs["_critic_input_size_delta"] = (
                kwargs["_critic_input_size_delta"] if "_critic_input_size_delta" in kwargs else 0
            ) - len(value_measure_adaptation)

        super().__init__(env, **kwargs)

        self._value_lambda = value_lambda
        self._value_loss_name = value_loss
        self._register_serializable("_value_lambda", "_value_loss_name")

        assert (
            self._value_loss_name in self.values_losses[self._critic_network_name]
        ), f"Value loss '{self._value_loss_name}' is not supported for network '{self._critic_network_name}'."
        value_loss_func = self.values_losses[critic_network][self._value_loss_name]
        self._value_loss = lambda *args, **kwargs: value_loss_func(self.critic, *args, **kwargs)

        if value_loss == self.value_loss_energy:
            value_loss_kwargs["sample_count"] = (
                value_loss_kwargs["sample_count"] if "sample_count" in value_loss_kwargs else 100
            )

        self._value_loss_kwargs = value_loss_kwargs
        self._register_serializable("_value_loss_kwargs")

        self._value_measure_adaptation = value_measure_adaptation
        self._register_serializable("_value_measure_adaptation")

        if self._critic_network_name == self.network_iqn:
            self._iqn_action_samples = iqn_action_samples
            self._iqn_value_samples = iqn_value_samples
            self._register_serializable("_iqn_action_samples", "_iqn_value_samples")

    def _critic_input(self, observations, actions=None) -> torch.Tensor:
        mask, shape = self._get_critic_obs_mask(observations)

        processed_observations = observations[mask].reshape(*shape)

        return processed_observations

    def _get_critic_obs_mask(self, observations):
        mask = torch.ones_like(observations).bool()

        if self._value_measure_adaptation is not None:
            mask[:, self._value_measure_adaptation] = False

        shape = (observations.shape[0], self._critic_input_size)

        return mask, shape

    def _process_quants(self, x):
        if self._value_loss_name == self.value_loss_energy:
            quants, idx = QuantileDistribution(x).sample(self._value_loss_kwargs["sample_count"])
        else:
            quants, idx = x, None

        return quants, idx

    @Benchmarkable.register
    def process_transition(self, *args) -> Dict[str, torch.Tensor]:
        transition = super(PPO, self).process_transition(*args)

        if self.recurrent:
            transition["critic_state_h"] = self.critic.hidden_state[0].detach()
            transition["critic_state_c"] = self.critic.hidden_state[1].detach()

        transition["full_critic_observations"] = transition["critic_observations"].detach()
        transition["full_next_critic_observations"] = transition["next_critic_observations"].detach()
        mask, shape = self._get_critic_obs_mask(transition["critic_observations"])
        transition["critic_observations"] = transition["critic_observations"][mask].reshape(*shape)
        transition["next_critic_observations"] = transition["next_critic_observations"][mask].reshape(*shape)

        critic_kwargs = (
            {"sample_count": self._iqn_action_samples} if self._critic_network_name == self.network_iqn else {}
        )
        transition["values"] = self.critic.forward(
            transition["critic_observations"],
            measure_args=self._extract_value_measure_adaptation(transition["full_critic_observations"]),
            **critic_kwargs,
        ).detach()

        if self._critic_network_name == self.network_iqn:
            # For IQN, we sample new (undistorted) quantiles for computing the value update
            critic_kwargs = (
                {"hidden_state": (transition["critic_state_h"], transition["critic_state_c"])} if self.recurrent else {}
            )
            self.critic.forward(
                transition["critic_observations"],
                sample_count=self._iqn_value_samples,
                use_measure=False,
                **critic_kwargs,
            ).detach()
            transition["value_taus"] = self.critic.last_taus.detach().reshape(transition["values"].shape[0], -1)

        transition["value_quants"] = self.critic.last_quantiles.detach().reshape(transition["values"].shape[0], -1)

        if self.recurrent:
            transition["critic_next_state_h"] = self.critic.hidden_state[0].detach()
            transition["critic_next_state_c"] = self.critic.hidden_state[1].detach()

        return transition

    @Benchmarkable.register
    def _compute_value_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        critic_kwargs = (
            {"sample_count": self._iqn_value_samples, "taus": batch["value_target_taus"], "use_measure": False}
            if self._critic_network_name == self.network_iqn
            else {}
        )

        if self.recurrent:
            observations, data = transitions_to_trajectories(batch["critic_observations"], batch["dones"])
            hidden_state_h, _ = transitions_to_trajectories(batch["critic_state_h"], batch["dones"])
            hidden_state_c, _ = transitions_to_trajectories(batch["critic_state_c"], batch["dones"])
            hidden_states = (hidden_state_h[0].transpose(0, 1), hidden_state_c[0].transpose(0, 1))

            if self._critic_network_name == self.network_iqn:
                critic_kwargs["taus"], _ = transitions_to_trajectories(critic_kwargs["taus"], batch["dones"])

            trajectory_evaluations = self.critic.forward(
                observations, distribution=True, hidden_state=hidden_states, **critic_kwargs
            )
            trajectory_evaluations = trajectory_evaluations.reshape(*observations.shape[:-1], -1)

            predictions = trajectories_to_transitions(trajectory_evaluations, data)
        else:
            predictions = self.critic.forward(batch["critic_observations"], distribution=True, **critic_kwargs)

        value_loss = self._value_loss(self._process_quants(predictions)[0], batch["value_target_quants"])

        return value_loss

    def _extract_value_measure_adaptation(self, observations: torch.Tensor) -> Tuple[torch.Tensor]:
        if self._value_measure_adaptation is None:
            return tuple()

        relevant_observations = observations[:, self._value_measure_adaptation]
        measure_adaptations = torch.tensor_split(relevant_observations, relevant_observations.shape[1], dim=1)

        return measure_adaptations

    @Benchmarkable.register
    def _process_dataset(self, dataset: Dataset) -> Dataset:
        rewards = torch.stack([entry["rewards"] for entry in dataset])
        dones = torch.stack([entry["dones"] for entry in dataset]).float()
        timeouts = torch.stack([entry["timeouts"] for entry in dataset])
        values = torch.stack([entry["values"] for entry in dataset])

        value_quants_idx = [self._process_quants(entry["value_quants"]) for entry in dataset]
        value_quants = torch.stack([entry[0] for entry in value_quants_idx])

        critic_kwargs = (
            {"hidden_state": (dataset[-1]["critic_state_h"], dataset[-1]["critic_state_c"])} if self.recurrent else {}
        )
        if self._critic_network_name == self.network_iqn:
            critic_kwargs["sample_count"] = self._iqn_action_samples

        measure_args = self._extract_value_measure_adaptation(dataset[-1]["full_next_critic_observations"])
        next_values = self.critic.forward(
            dataset[-1]["next_critic_observations"], measure_args=measure_args, **critic_kwargs
        )

        if self._critic_network_name == self.network_iqn:
            # For IQN, we sample new (undistorted) quantiles for computing the value update
            critic_kwargs["sample_count"] = self._iqn_value_samples
            self.critic.forward(
                dataset[-1]["next_critic_observations"],
                use_measure=False,
                **critic_kwargs,
            )

            final_value_taus = self.critic.last_taus
            value_taus = torch.stack(
                [
                    torch.take_along_dim(dataset[i]["value_taus"], value_quants_idx[i][1], -1)
                    for i in range(len(dataset))
                ]
            )

        final_value_quants = self.critic.last_quantiles

        # Timeout bootstrapping for rewards.
        rewards += self.gamma * timeouts * values

        # Compute advantages and value target quantiles
        next_values = torch.cat((values[1:], next_values.unsqueeze(0)), dim=0)
        deltas = (rewards + (1 - dones) * self.gamma * next_values - values).reshape(-1, self.env.num_envs)
        advantages = torch.zeros((len(dataset) + 1, self.env.num_envs), device=self.device)

        next_value_quants, idx = self._process_quants(final_value_quants)
        value_target_quants = torch.zeros(len(dataset), *next_value_quants.shape, device=self.device)

        if self._critic_network_name == self.network_iqn:
            value_target_taus = torch.zeros(len(dataset) + 1, *next_value_quants.shape, device=self.device)
            value_target_taus[-1] = torch.take_along_dim(final_value_taus, idx, -1)

        for step in reversed(range(len(dataset))):
            not_terminal = 1.0 - dones[step]
            not_terminal_ = not_terminal.unsqueeze(-1)

            advantages[step] = deltas[step] + (1.0 - dones[step]) * self.gamma * self._gae_lambda * advantages[step + 1]
            value_target_quants[step] = rewards[step].unsqueeze(-1) + not_terminal_ * self.gamma * next_value_quants

            preserved_value_quants = not_terminal_.bool() * (
                torch.rand(next_value_quants.shape, device=self.device) < self._value_lambda
            )
            next_value_quants = torch.where(preserved_value_quants, value_target_quants[step], value_quants[step])

            if self._critic_network_name == self.network_iqn:
                value_target_taus[step] = torch.where(
                    preserved_value_quants, value_target_taus[step + 1], value_taus[step]
                )

        advantages = advantages[:-1]
        if self._critic_network_name == self.network_iqn:
            value_target_taus = value_target_taus[:-1]

        # Normalize advantages and pack into dataset structure.
        amean, astd = advantages.mean(), torch.nan_to_num(advantages.std())
        for step in range(len(dataset)):
            dataset[step]["advantages"] = advantages[step]
            dataset[step]["normalized_advantages"] = (advantages[step] - amean) / (astd + 1e-8)
            dataset[step]["value_target_quants"] = value_target_quants[step]

            if self._critic_network_name == self.network_iqn:
                dataset[step]["value_target_taus"] = value_target_taus[step]

        return dataset
