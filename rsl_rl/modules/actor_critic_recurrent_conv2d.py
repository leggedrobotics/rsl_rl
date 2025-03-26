# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class ActorCriticRecurrentConv2d(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        image_input_shape,
        conv_layers_params,
        conv_linear_output_size,
        actor_hidden_dims,
        critic_hidden_dims,
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        super().__init__()

        self.image_input_shape = image_input_shape  # (C, H, W) [3, 100, 100]
        self.activation_fn = resolve_nn_activation(activation)

        self.actor = ConvolutionalNetwork(
            proprio_input_dim=num_actor_obs,
            output_dim=num_actions,
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            hidden_dims=actor_hidden_dims,
            activation_fn=self.activation_fn,
            conv_linear_output_size=conv_linear_output_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers
        )

        self.critic = ConvolutionalNetwork(
            proprio_input_dim=num_critic_obs,
            output_dim=1,
            image_input_shape=image_input_shape,
            conv_layers_params=conv_layers_params,
            hidden_dims=critic_hidden_dims,
            activation_fn=self.activation_fn,
            conv_linear_output_size=conv_linear_output_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers
        )

        print(f"Modified Actor Network: {self.actor}")
        print(f"Modified Critic Network: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        # Reset hidden states when episodes terminate
        self.actor.reset(dones)
        self.critic.reset(dones)

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

    def update_distribution(self, observations, masks=None, hidden_states=None):
        mean = self.actor(observations, masks=masks, hidden_states=hidden_states)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, masks=None, hidden_states=None):  # [512, 30055]
        self.update_distribution(observations, masks=masks, hidden_states=hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # critic_obs: [512, 30055]
        value = self.critic(critic_observations, masks=masks, hidden_states=hidden_states).unsqueeze(1)
        return value
    
    def get_hidden_states(self):
        return self.actor.hidden_states, self.critic.hidden_states
    


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
        rnn_hidden_size,
        rnn_num_layers
    ):
        super().__init__()

        self.image_input_shape = image_input_shape  # (C, H, W)
        self.image_obs_size = torch.prod(torch.tensor(self.image_input_shape)).item()
        self.proprio_obs_size = proprio_input_dim
        self.input_dim = self.proprio_obs_size + self.image_obs_size
        self.activation_fn = activation_fn

        # Build conv network and get its output size
        self.conv_net = self.build_conv_net(conv_layers_params)
        with torch.no_grad():
            dummy_image = torch.zeros(1, *self.image_input_shape)
            conv_output = self.conv_net(dummy_image)            
            self.image_feature_size = conv_output.view(1, -1).shape[1]

        # Build the connection layers between conv net and mlp -> flattened into 1D vector
        self.conv_linear = nn.Linear(self.image_feature_size, conv_linear_output_size)
        self.layernorm = nn.LayerNorm(conv_linear_output_size)

        # Build the LSTM layer
        self.rnn = nn.LSTM(
            input_size=conv_linear_output_size + proprio_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers
        )
        self.hidden_states = None
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers


        # Build the mlp
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_dims[0]),
            self.activation_fn,
            *[
                layer
                for dim in zip(hidden_dims[:-1], hidden_dims[1:])
                for layer in (nn.Linear(dim[0], dim[1]), self.activation_fn)
            ],
            nn.Linear(hidden_dims[-1], output_dim),
        )

        # Initialize the weights
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

    def forward(self, observations, masks=None, hidden_states=None):
        batch_mode = masks is not None
        
        # Pass through LSTM
        if batch_mode:
            # obs: [24, 131, 30055]
            proprio_obs = observations[:, :, : -self.image_obs_size]    #[24, 131, 55]
            num_proprio = proprio_obs.shape[-1]
            image_obs = observations[:, :, -self.image_obs_size :]      # [24, 131, 30000]
            batch_size, sequence_length, _ = image_obs.shape
            image = image_obs.reshape(-1, *self.image_input_shape)  #[3144, 3, 100, 100]

            conv_features = self.conv_net(image)                    #[3144, 16, 11, 11]
            flattened_conv_features = conv_features.view(batch_size * sequence_length, -1) # [3144, 1936]
            normalized_conv_output = self.layernorm(self.conv_linear(flattened_conv_features))
            
            proprio_obs = proprio_obs.reshape(-1, num_proprio)  # [24 * 131, 55]
            
            combined_input = torch.cat([proprio_obs, normalized_conv_output], dim=1)
            combined_input = combined_input.view(batch_size, sequence_length, -1)

            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            
            # input: [24, 131, 71]
            lstm_out, _ = self.rnn(combined_input, hidden_states)
            lstm_out = unpad_trajectories(lstm_out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            proprio_obs = observations[:, : -self.image_obs_size]       # [512, 55]
            image_obs = observations[:, -self.image_obs_size :]         # [512, 30000]
            
            batch_size = image_obs.size(0)
            image = image_obs.view(batch_size, *self.image_input_shape) #[512, 3, 100, 100]
            
            conv_features = self.conv_net(image)    # [512, 16, 11, 11]
            flattened_conv_features = conv_features.view(batch_size, -1) #[512, 1936]
            normalized_conv_output = self.layernorm(self.conv_linear(flattened_conv_features)) #[512, 16]
            combined_input = torch.cat([proprio_obs, normalized_conv_output], dim=1) # Shape: [512, 71]

            lstm_out, self.hidden_states = self.rnn(combined_input.unsqueeze(0), self.hidden_states)
        
        # lstm_out: [1, 512, 256]
        output = self.mlp(lstm_out).squeeze()
        return output
    

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if dones is None or self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            # Size: [1, 512, 256]
            hidden_state[..., dones == 1, :] = 0.0



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