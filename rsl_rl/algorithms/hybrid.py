from abc import ABC, abstractmethod
import torch
from typing import Callable, Dict, Tuple, Type, Union

from rsl_rl.algorithms import D4PG, DSAC
from rsl_rl.algorithms import TD3
from rsl_rl.algorithms.actor_critic import AbstractActorCritic
from rsl_rl.algorithms.agent import Agent
from rsl_rl.env import VecEnv
from rsl_rl.storage.storage import Dataset, Storage


class AbstractHybridAgent(Agent, ABC):
    def __init__(
        self,
        env: VecEnv,
        agent_class: Type[Agent],
        agent_kwargs: dict,
        pretrain_agent_class: Type[Agent],
        pretrain_agent_kwargs: dict,
        pretrain_steps: int,
        freeze_steps: int = 0,
        **general_kwargs,
    ):
        """
        Args:
            env (VecEnv): A vectorized environment.
        """
        agent_kwargs["env"] = env
        pretrain_agent_kwargs["env"] = env

        self.agent = agent_class(**agent_kwargs, **general_kwargs)
        self.pretrain_agent = pretrain_agent_class(**pretrain_agent_kwargs, **general_kwargs)

        self._freeze_steps = freeze_steps
        self._pretrain_steps = pretrain_steps
        self._steps = 0

        self._register_serializable("agent", "pretrain_agent", "_freeze_steps", "_pretrain_steps", "_steps")

    @property
    def active_agent(self):
        agent = self.pretrain_agent if self.pretraining else self.agent

        return agent

    def draw_actions(self, *args, **kwargs) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        return self.active_agent.draw_actions(*args, **kwargs)

    def draw_random_actions(self, *args, **kwargs) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        return self.active_agent.draw_random_actions(*args, **kwargs)

    def eval_mode(self, *args, **kwargs) -> Agent:
        self.agent.eval_mode(*args, **kwargs)

    def get_inference_policy(self, *args, **kwargs) -> Callable:
        return self.active_agent.get_inference_policy(*args, **kwargs)

    @property
    def initialized(self) -> bool:
        return self.active_agent.initialized

    @property
    def pretraining(self):
        return self._steps < self._pretrain_steps

    def process_dataset(self, *args, **kwargs) -> Dataset:
        return self.active_agent.process_dataset(*args, **kwargs)

    def process_transition(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.active_agent.process_transition(*args, **kwargs)

    def register_terminations(self, *args, **kwargs) -> None:
        return self.active_agent.register_terminations(*args, **kwargs)

    @property
    def storage(self) -> Storage:
        return self.active_agent.storage

    def to(self, *args, **kwargs) -> Agent:
        self.agent.to(*args, **kwargs)
        self.pretrain_agent.to(*args, **kwargs)

    def train_mode(self, *args, **kwargs) -> Agent:
        self.agent.train_mode(*args, **kwargs)
        self.pretrain_agent.train_mode(*args, **kwargs)

    def update(self, *args, **kwargs) -> Dict[str, Union[float, torch.Tensor]]:
        result = self.active_agent.update(*args, **kwargs)

        if not self.active_agent.initialized:
            return

        self._steps += 1

        if self._steps == self._pretrain_steps:
            self._transfer_weights()
            self._freeze_weights(freeze=True)

        if self._steps == self._pretrain_steps + self._freeze_steps:
            self._transfer_weights()
            self._freeze_weights(freeze=False)

        return result

    @abstractmethod
    def _freeze_weights(self, freeze=True):
        pass

    @abstractmethod
    def _transfer_weights(self):
        pass


class HybridD4PG(AbstractHybridAgent):
    def __init__(
        self,
        env: VecEnv,
        d4pg_kwargs: dict,
        pretrain_kwargs: dict,
        pretrain_agent: Type[AbstractActorCritic] = TD3,
        **kwargs,
    ):
        assert d4pg_kwargs["action_max"] == pretrain_kwargs["action_max"]
        assert d4pg_kwargs["action_min"] == pretrain_kwargs["action_min"]
        assert d4pg_kwargs["actor_activations"] == pretrain_kwargs["actor_activations"]
        assert d4pg_kwargs["actor_hidden_dims"] == pretrain_kwargs["actor_hidden_dims"]
        assert d4pg_kwargs["actor_input_normalization"] == pretrain_kwargs["actor_input_normalization"]

        super().__init__(
            env,
            agent_class=D4PG,
            agent_kwargs=d4pg_kwargs,
            pretrain_agent_class=pretrain_agent,
            pretrain_agent_kwargs=pretrain_kwargs,
            **kwargs,
        )

    def _freeze_weights(self, freeze=True):
        for param in self.agent.actor.parameters():
            param.requires_grad = not freeze

    def _transfer_weights(self):
        self.agent.actor.load_state_dict(self.pretrain_agent.actor.state_dict())
        self.agent.actor_optimizer.load_state_dict(self.pretrain_agent.actor_optimizer.state_dict())


class HybridDSAC(AbstractHybridAgent):
    def __init__(
        self,
        env: VecEnv,
        dsac_kwargs: dict,
        pretrain_kwargs: dict,
        pretrain_agent: Type[AbstractActorCritic] = TD3,
        **kwargs,
    ):
        assert dsac_kwargs["action_max"] == pretrain_kwargs["action_max"]
        assert dsac_kwargs["action_min"] == pretrain_kwargs["action_min"]
        assert dsac_kwargs["actor_activations"] == pretrain_kwargs["actor_activations"]
        assert dsac_kwargs["actor_hidden_dims"] == pretrain_kwargs["actor_hidden_dims"]
        assert dsac_kwargs["actor_input_normalization"] == pretrain_kwargs["actor_input_normalization"]

        super().__init__(
            env,
            agent_class=DSAC,
            agent_kwargs=dsac_kwargs,
            pretrain_agent_class=pretrain_agent,
            pretrain_agent_kwargs=pretrain_kwargs,
            **kwargs,
        )

    def _freeze_weights(self, freeze=True):
        """Freezes actor layers.

        Freezes feature encoding and mean computation layers for gaussian network. Leaves log standard deviation layer
        unfreezed.
        """
        for param in self.agent.actor._layers.parameters():
            param.requires_grad = not freeze

        for param in self.agent.actor._mean_layer.parameters():
            param.requires_grad = not freeze

    def _transfer_weights(self):
        """Transfers actor layers.

        Transfers only feature encoding and mean computation layers for gaussian network.
        """
        for i, layer in enumerate(self.agent.actor._layers):
            layer.load_state_dict(self.pretrain_agent.actor._layers[i].state_dict())

        for j, layer in enumerate(self.agent.actor._mean_layer):
            layer.load_state_dict(self.pretrain_agent.actor._layers[i + j + 1].state_dict())
