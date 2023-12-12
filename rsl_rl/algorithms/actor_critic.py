from abc import abstractmethod
import torch
from typing import Any, Callable, Dict, List, Tuple, Union

from rsl_rl.algorithms.agent import Agent
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.modules.network import Network
from rsl_rl.storage.storage import Dataset
from rsl_rl.utils.utils import environment_dimensions
from rsl_rl.utils.utils import squeeze_preserve_batch


class AbstractActorCritic(Agent):
    _alg_features = dict(recurrent=False)

    def __init__(
        self,
        env: VecEnv,
        actor_activations: List[str] = ["relu", "relu", "relu", "linear"],
        actor_hidden_dims: List[int] = [256, 256, 256],
        actor_init_gain: float = 0.5,
        actor_input_normalization: bool = False,
        actor_recurrent_layers: int = 1,
        actor_recurrent_module: str = Network.recurrent_module_lstm,
        actor_recurrent_tf_context_length: int = 64,
        actor_recurrent_tf_head_count: int = 8,
        actor_shared_dims: int = None,
        batch_count: int = 1,
        batch_size: int = 1,
        critic_activations: List[str] = ["relu", "relu", "relu", "linear"],
        critic_hidden_dims: List[int] = [256, 256, 256],
        critic_init_gain: float = 0.5,
        critic_input_normalization: bool = False,
        critic_recurrent_layers: int = 1,
        critic_recurrent_module: str = Network.recurrent_module_lstm,
        critic_recurrent_tf_context_length: int = 64,
        critic_recurrent_tf_head_count: int = 8,
        critic_shared_dims: int = None,
        polyak: float = 0.995,
        recurrent: bool = False,
        return_steps: int = 1,
        _actor_input_size_delta: int = 0,
        _critic_input_size_delta: int = 0,
        **kwargs,
    ):
        """Creates an actor critic agent.

        Args:
            env (VecEnv): A vectorized environment.
            actor_activations (List[str]): A list of activation functions for the actor network.
            actor_hidden_dims (List[str]): A list of layer sizes for the actor network.
            actor_init_gain (float): Network initialization gain for actor.
            actor_input_normalization (bool): Whether to empirically normalize inputs to the actor network.
            actor_recurrent_layers (int): The number of recurrent layers to use for the actor network.
            actor_recurrent_module (str): The recurrent module to use. Must be one of Network.recurrent_modules.
            actor_shared_dims (int): The number of dimensions to share for an actor with multiple heads.
            batch_count (int): The number of batches to process per update step.
            batch_size (int): The size of each batch to process during the update step.
            critic_activations (List[str]): A list of activation functions for the critic network.
            critic_hidden_dims: (List[str]): A list of layer sizes for the critic network.
            critic_init_gain (float): Network initialization gain for critic.
            critic_input_normalization (bool): Whether to empirically normalize inputs to the critic network.
            critic_recurrent_layers (int): The number of recurrent layers to use for the critic network.
            critic_recurrent_module (str): The recurrent module to use. Must be one of Network.recurrent_modules.
            critic_shared_dims (int): The number of dimensions to share for a critic with multiple heads.
            polyak (float): The actor-critic target network polyak factor.
            recurrent (bool): Whether to use recurrent actor and critic networks.
            recurrent_module (str): The recurrent module to use. Must be one of Network.recurrent_modules.
            recurrent_tf_context_length (int): The context length of the Transformer.
            recurrent_tf_head_count (int): The head count of the Transformer.
            return_steps (float): The number of steps over which to compute the returns (n-step return).
            _actor_input_size_delta (int): The number of additional dimensions to add to the actor input.
            _critic_input_size_delta (int): The number of additional dimensions to add to the critic input.
        """
        assert (
            self._alg_features["recurrent"] == True or not recurrent
        ), f"{self.__class__.__name__} does not support recurrent networks."

        super().__init__(env, **kwargs)

        self.actor: torch.nn.Module = None
        self.actor_optimizer: torch.nn.Module = None
        self.critic_optimizer: torch.nn.Module = None
        self.critic: torch.nn.Module = None

        self._batch_size = batch_size
        self._batch_count = batch_count
        self._polyak_factor = polyak
        self._return_steps = return_steps
        self._recurrent = recurrent

        self._register_serializable(
            "_batch_size", "_batch_count", "_discount_factor", "_polyak_factor", "_return_steps"
        )

        dimensions = environment_dimensions(self.env)
        try:
            actor_input_size = dimensions["actor_observations"]
            critic_input_size = dimensions["critic_observations"]
        except KeyError:
            actor_input_size = dimensions["observations"]
            critic_input_size = dimensions["observations"]
        self._actor_input_size = actor_input_size + _actor_input_size_delta
        self._critic_input_size = critic_input_size + self._action_size + _critic_input_size_delta

        self._register_actor_network_kwargs(
            activations=actor_activations,
            hidden_dims=actor_hidden_dims,
            init_gain=actor_init_gain,
            input_normalization=actor_input_normalization,
            recurrent=recurrent,
            recurrent_layers=actor_recurrent_layers,
            recurrent_module=actor_recurrent_module,
            recurrent_tf_context_length=actor_recurrent_tf_context_length,
            recurrent_tf_head_count=actor_recurrent_tf_head_count,
        )

        if actor_shared_dims is not None:
            self._register_actor_network_kwargs(shared_dims=actor_shared_dims)

        self._register_critic_network_kwargs(
            activations=critic_activations,
            hidden_dims=critic_hidden_dims,
            init_gain=critic_init_gain,
            input_normalization=critic_input_normalization,
            recurrent=recurrent,
            recurrent_layers=critic_recurrent_layers,
            recurrent_module=critic_recurrent_module,
            recurrent_tf_context_length=critic_recurrent_tf_context_length,
            recurrent_tf_head_count=critic_recurrent_tf_head_count,
        )

        if critic_shared_dims is not None:
            self._register_critic_network_kwargs(shared_dims=critic_shared_dims)

        self._register_serializable(
            "_actor_input_size", "_actor_network_kwargs", "_critic_input_size", "_critic_network_kwargs"
        )

        # For computing n-step returns using prior transitions.
        self._stored_dataset = []

    def export_onnx(self) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
        self.eval_mode()

        class ONNXActor(torch.nn.Module):
            def __init__(self, model: torch.nn.Module):
                super().__init__()

                self.model = model

            def forward(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor] = None):
                if hidden_state is None:
                    return self.model(x)

                data = self.model(x, hidden_state=hidden_state)
                hidden_state = self.model.last_hidden_state

                return data, hidden_state

        model = ONNXActor(self.actor)
        kwargs = dict(
            export_params=True,
            opset_version=11,
            verbose=True,
            dynamic_axes={},
        )

        kwargs["input_names"] = ["observations"]
        kwargs["output_names"] = ["actions"]

        args = torch.zeros(1, self._actor_input_size)

        if self.actor.recurrent:
            hidden_state = (
                torch.zeros(self.actor._features[0].num_layers, 1, self.actor._features[0].hidden_size),
                torch.zeros(self.actor._features[0].num_layers, 1, self.actor._features[0].hidden_size),
            )
            args = (args, {"hidden_state": hidden_state})

        return model, args, kwargs

    def draw_random_actions(self, obs: torch.Tensor, env_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict]:
        actions, data = super().draw_random_actions(obs, env_info)

        actor_obs, critic_obs = self._process_observations(obs, env_info)
        data.update({"actor_observations": actor_obs.clone(), "critic_observations": critic_obs.clone()})

        return actions, data

    def get_inference_policy(self, device=None) -> Callable:
        self.to(device)
        self.eval_mode()

        if self.actor.recurrent:
            self.actor.reset_full_hidden_state(batch_size=self.env.num_envs)

        if self.critic.recurrent:
            self.critic.reset_full_hidden_state(batch_size=self.env.num_envs)

        def policy(obs, env_info=None):
            with torch.inference_mode():
                obs, _ = self._process_observations(obs, env_info)

                actions = self._process_actions(self.actor.forward(obs))

            return actions

        return policy

    def process_transition(
        self,
        observations: torch.Tensor,
        environment_info: Dict[str, Any],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_environment_info: torch.Tensor,
        dones: torch.Tensor,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if "actor_observations" in data and "critic_observations" in data:
            actor_obs, critic_obs = data["actor_observations"], data["critic_observations"]
        else:
            actor_obs, critic_obs = self._process_observations(observations, environment_info)

        if "next_actor_observations" in data and "next_critic_observations" in data:
            next_actor_obs, next_critic_obs = data["next_actor_observations"], data["next_critic_observations"]
        else:
            next_actor_obs, next_critic_obs = self._process_observations(next_observations, next_environment_info)

        transition = {
            "actions": actions,
            "actor_observations": actor_obs,
            "critic_observations": critic_obs,
            "dones": dones,
            "next_actor_observations": next_actor_obs,
            "next_critic_observations": next_critic_obs,
            "rewards": squeeze_preserve_batch(rewards),
            "timeouts": self._extract_timeouts(next_environment_info),
        }
        transition.update(data)

        for key, value in transition.items():
            transition[key] = value.detach().clone()

        return transition

    @property
    def recurrent(self) -> bool:
        return self._recurrent

    def register_terminations(self, terminations: torch.Tensor) -> None:
        pass

    @abstractmethod
    def update(self, dataset: Dataset) -> Dict[str, Union[float, torch.Tensor]]:
        with torch.inference_mode():
            self.storage.append(self._process_dataset(dataset))

    def _critic_input(self, observations, actions) -> torch.Tensor:
        """Combines observations and actions into a tensor that can be fed into the critic network.

        Args:
            observations (torch.Tensor): The critic observations.
            actions (torch.Tensor): The actions computed by the actor.
        Returns:
            A torch.Tensor of inputs for the critic network.
        """
        return torch.cat((observations, actions), dim=-1)

    def _extract_timeouts(self, next_environment_info):
        """Extracts timeout information from the transition next state information dictionary.

        Args:
            next_environment_info (Dict[str, Any]): The transition next state information dictionary.
        Returns:
            A torch.Tensor vector of actor timeouts.
        """
        if "time_outs" not in next_environment_info:
            return torch.zeros(self.env.num_envs, device=self.device)

        timeouts = squeeze_preserve_batch(next_environment_info["time_outs"].to(self.device))

        return timeouts

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Processes a dataset before it is added to the replay memory.

        Handles n-step returns and timeouts.

        TODO: This function seems to be a bottleneck in the training pipeline - speed it up!

        Args:
            dataset (Dataset): The dataset to process.
        Returns:
            A Dataset object containing the processed data.
        """
        assert len(dataset) >= self._return_steps

        dataset = self._stored_dataset + dataset
        length = len(dataset) - self._return_steps + 1
        self._stored_dataset = dataset[length:]

        for idx in range(len(dataset) - self._return_steps + 1):
            dead_idx = torch.zeros_like(dataset[idx]["dones"])
            rewards = torch.zeros_like(dataset[idx]["rewards"])

            for k in range(self._return_steps):
                data = dataset[idx + k]
                alive_idx = (dead_idx == 0).nonzero()
                critic_predictions = self.critic.forward(
                    self._critic_input(
                        data["critic_observations"].clone().to(self.device),
                        data["actions"].clone().to(self.device),
                    )
                )
                rewards[alive_idx] += self._discount_factor**k * data["rewards"][alive_idx]
                rewards[alive_idx] += (
                    self._discount_factor ** (k + 1) * data["timeouts"][alive_idx] * critic_predictions[alive_idx]
                )
                dead_idx += data["dones"]
                dead_idx += data["timeouts"]

            dataset[idx]["rewards"] = rewards

        return dataset[:length]

    def _process_observations(
        self, observations: torch.Tensor, environment_info: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Processes observations returned by the environment to extract actor and critic observations.

        Args:
            observations (torch.Tensor): normal environment observations.
            environment_info (Dict[str, Any]): A dictionary of additional environment information.
        Returns:
            A tuple containing two torch.Tensors with actor and critic observations, respectively.
        """
        try:
            critic_obs = environment_info["observations"]["critic"]
        except (KeyError, TypeError):
            critic_obs = observations

        actor_obs, critic_obs = observations.to(self.device), critic_obs.to(self.device)

        return actor_obs, critic_obs

    def _register_actor_network_kwargs(self, **kwargs) -> None:
        """Function to configure actor network in child classes before calling super().__init__()."""
        if not hasattr(self, "_actor_network_kwargs"):
            self._actor_network_kwargs = dict()

        self._actor_network_kwargs.update(**kwargs)

    def _register_critic_network_kwargs(self, **kwargs) -> None:
        """Function to configure critic network in child classes before calling super().__init__()."""
        if not hasattr(self, "_critic_network_kwargs"):
            self._critic_network_kwargs = dict()

        self._critic_network_kwargs.update(**kwargs)

    def _update_target(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        """Updates the target network using the polyak factor.

        Args:
            online (torch.nn.Module): The online network.
            target (torch.nn.Module): The target network.
        """
        for op, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_((1.0 - self._polyak_factor) * op.data + self._polyak_factor * tp.data)
