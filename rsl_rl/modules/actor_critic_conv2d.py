# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        proprio_input_dim,
        output_dim,
        image_input_shape,
        conv_layers_params,
        hidden_dims,
        activation_fn,
        conv_linear_output_size,
    ):
        super().__init__()

        self.image_input_shape = image_input_shape  # (C, H, W)
        self.image_obs_size = torch.prod(torch.tensor(self.image_input_shape)).item()
        self.proprio_obs_size = proprio_input_dim
        self.input_dim = self.proprio_obs_size + self.image_obs_size
        self.activation_fn = activation_fn

        # build conv network and get its output size
        self.conv_net = self.build_conv_net(conv_layers_params)
        with torch.no_grad():
            dummy_image = torch.zeros(1, *self.image_input_shape)
            conv_output = self.conv_net(dummy_image)
            self.image_feature_size = conv_output.view(1, -1).shape[1]

        # connection layers between conv net and mlp
        self.conv_linear = nn.Linear(self.image_feature_size, conv_linear_output_size)
        self.layernorm = nn.LayerNorm(conv_linear_output_size)

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.proprio_obs_size + conv_linear_output_size, hidden_dims[0]),
            self.activation_fn,
            *[
                layer
                for dim in zip(hidden_dims[:-1], hidden_dims[1:])
                for layer in (nn.Linear(dim[0], dim[1]), self.activation_fn)
            ],
            nn.Linear(hidden_dims[-1], output_dim),
        )

        # initialize weights
        self._initialize_weights()

    def build_conv_net(self, conv_layers_params):
        layers = []
        in_channels = self.image_input_shape[0]
        for idx, params in enumerate(conv_layers_params[:-1]):
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    params["out_channels"],
                    kernel_size=params.get("kernel_size", 3),
                    stride=params.get("stride", 1),
                    padding=params.get("padding", 0),
                ),
                nn.BatchNorm2d(params["out_channels"]),
                nn.ReLU(inplace=True),
                ResidualBlock(params["out_channels"]) if idx > 0 else nn.Identity(),
            ])
            in_channels = params["out_channels"]
        last_params = conv_layers_params[-1]
        layers.append(
            nn.Conv2d(
                in_channels,
                last_params["out_channels"],
                kernel_size=last_params.get("kernel_size", 3),
                stride=last_params.get("stride", 1),
                padding=last_params.get("padding", 0),
            )
        )
        layers.append(nn.BatchNorm2d(last_params["out_channels"]))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.conv_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.conv_linear.weight, mode="fan_out", nonlinearity="tanh")
        nn.init.constant_(self.conv_linear.bias, 0)
        nn.init.constant_(self.layernorm.weight, 1.0)
        nn.init.constant_(self.layernorm.bias, 0.0)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias) if layer.bias is not None else None

    def forward(self, proprio_obs, image_obs):
        conv_features = self.conv_net(image_obs)
        flattened_conv_features = conv_features.reshape(conv_features.size(0), -1)
        normalized_conv_output = self.layernorm(self.conv_linear(flattened_conv_features))
        combined_input = torch.cat([proprio_obs, normalized_conv_output], dim=1)
        output = self.mlp(combined_input)
        return output


class ActorCriticConv2d(nn.Module):
    is_recurrent = False
    is_conv2d = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        conv_layers_params,
        conv_linear_output_size,
        actor_hidden_dims,
        critic_hidden_dims,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticConv2d.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.activation_fn = resolve_nn_activation(activation)

        # get observation dimensions
        self.num_actor_obs, self.actor_image_shape = self._calculate_obs_dims(obs, obs_groups["policy"])
        self.num_critic_obs, self.critic_image_shape = self._calculate_obs_dims(obs, obs_groups["critic"])
        
        self.image_input_shape = self.actor_image_shape
        if self.image_input_shape is None:
            raise ValueError("No image observations found. Conv2d networks require image inputs.")

        # actor
        self.actor = ConvolutionalNetwork(
            proprio_input_dim=self.num_actor_obs,
            output_dim=num_actions,
            image_input_shape=self.image_input_shape,
            conv_layers_params=conv_layers_params,
            hidden_dims=actor_hidden_dims,
            activation_fn=self.activation_fn,
            conv_linear_output_size=conv_linear_output_size,
        )

        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # critic
        self.critic = ConvolutionalNetwork(
            proprio_input_dim=self.num_critic_obs,
            output_dim=1,
            image_input_shape=self.image_input_shape,
            conv_layers_params=conv_layers_params,
            hidden_dims=critic_hidden_dims,
            activation_fn=self.activation_fn,
            conv_linear_output_size=conv_linear_output_size,
        )

        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        print(f"Actor ConvNet: {self.actor}")
        print(f"Critic ConvNet: {self.critic}")

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # action distribution
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _calculate_obs_dims(self, obs, obs_group_list):
        """Calculate total proprioceptive obs dim and extract image shape."""
        total_proprio_dim = 0
        image_shape = None
        
        for obs_group in obs_group_list:
            obs_tensor = obs[obs_group]
            if obs_group == "sensor":
                image_shape = obs_tensor.permute(0, 3, 1, 2).shape[1:]
            else:
                assert len(obs_tensor.shape) == 2, f"Non-image observations must be 1D. Got {obs_tensor.shape} for {obs_group}"
                total_proprio_dim += obs_tensor.shape[-1]
        
        return total_proprio_dim, image_shape

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, proprio_obs, image_obs):
        mean = self.actor(proprio_obs, image_obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        proprio_obs, image_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        self.update_distribution(proprio_obs, image_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        proprio_obs, image_obs = self.get_actor_obs(obs)
        proprio_obs = self.actor_obs_normalizer(proprio_obs)
        return self.actor(proprio_obs, image_obs)

    def evaluate(self, obs, **kwargs):
        proprio_obs, image_obs = self.get_critic_obs(obs)
        proprio_obs = self.critic_obs_normalizer(proprio_obs)
        return self.critic(proprio_obs, image_obs)

    def get_actor_obs(self, obs):
        obs_list = []
        image_obs = None
        
        for obs_group in self.obs_groups["policy"]:
            if obs_group == "sensor":
                image_obs = obs[obs_group].permute(0, 3, 1, 2)
            else:
                obs_list.append(obs[obs_group])
        
        if obs_list:
            proprio_obs = torch.cat(obs_list, dim=-1)
        else:
            proprio_obs = torch.zeros(obs[list(obs.keys())[0]].shape[0], 0, device=obs.device)
        
        if image_obs is not None:
            return proprio_obs, image_obs
        else:
            dummy_image = torch.zeros(proprio_obs.shape[0], *self.image_input_shape, device=proprio_obs.device)
            return proprio_obs, dummy_image

    def get_critic_obs(self, obs):
        obs_list = []
        image_obs = None
        
        for obs_group in self.obs_groups["critic"]:
            if obs_group == "sensor":
                image_obs = obs[obs_group].permute(0, 3, 1, 2)
            else:
                obs_list.append(obs[obs_group])
        
        if obs_list:
            proprio_obs = torch.cat(obs_list, dim=-1)
        else:
            proprio_obs = torch.zeros(obs[list(obs.keys())[0]].shape[0], 0, device=obs.device)
        
        if image_obs is not None:
            return proprio_obs, image_obs
        else:
            dummy_image = torch.zeros(proprio_obs.shape[0], *self.image_input_shape, device=proprio_obs.device)
            return proprio_obs, dummy_image

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            proprio_obs, _ = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(proprio_obs)
        if self.critic_obs_normalization:
            proprio_obs, _ = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(proprio_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
