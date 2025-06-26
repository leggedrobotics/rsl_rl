from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from robot_rl.modules import ActorCritic
from robot_rl.utils import resolve_nn_activation


class ActorCriticEstimator(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_estimates,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        estimator_index=-1,
        oracle=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.num_estimates = num_estimates
        if estimator_index == -1:
            estimator_index = len(actor_hidden_dims) - 1
        # Policy
        shared_layers = []
        actor_layers = []
        shared_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        shared_layers.append(resolve_nn_activation(activation))
        for layer_index in range(len(actor_hidden_dims) - 1):
            if layer_index < estimator_index:
                shared_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                shared_layers.append(resolve_nn_activation(activation))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(resolve_nn_activation(activation))
        self.backbone = nn.Sequential(*shared_layers)
        self.actor = nn.Sequential(*shared_layers, *actor_layers, nn.Linear(actor_hidden_dims[-1], num_actions))

        if oracle:
            oracle_layers = []
            oracle_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            oracle_layers.append(resolve_nn_activation(activation))
            for layer_index in range(len(actor_hidden_dims) - 1):
                    oracle_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                    oracle_layers.append(resolve_nn_activation(activation))
            self.estimator = nn.Sequential(*oracle_layers, nn.Linear(actor_hidden_dims[-1], num_estimates))
        else:
            self.estimator = nn.Sequential(*shared_layers, nn.Linear(actor_hidden_dims[-1], num_estimates))


        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(resolve_nn_activation(activation))
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(resolve_nn_activation(activation))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Estimator MLP: {self.estimator}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def estimate(self, observations, **kwargs) -> torch.Tensor:
        estimates = self.estimator(observations)
        return estimates
    
    def get_latents(self, observations, **kwargs) -> torch.Tensor:
        latents = self.backbone(observations)
        return latents
    
    def get_last_layer(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Get the weight matrix of the last layer of the actor
        actor_last_layer = self.actor[-1]
        actor_weights = actor_last_layer.weight if isinstance(actor_last_layer, nn.Linear) else None

        # Get the weight matrix of the last layer of the estimator
        estimator_last_layer = self.estimator[-1]
        estimator_weights = estimator_last_layer.weight if isinstance(estimator_last_layer, nn.Linear) else None

        return actor_weights, estimator_weights
