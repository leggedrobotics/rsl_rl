from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from robot_rl.utils import resolve_nn_activation

class MHAEncoder(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_latent: int,
        n_heads: int,
        n_channels: int,
        kernel_size: int,
        dropout: float = 0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.proprio_enc = nn.Sequential(
            nn.Linear(n_obs, n_latent),
            resolve_nn_activation(activation),
        )
        self.height_enc = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size, padding="same"),
            resolve_nn_activation(activation),
            nn.Conv2d(n_channels, n_latent-3, kernel_size, padding="same"),
            resolve_nn_activation(activation),
        )
        self.mha = nn.MultiheadAttention(n_latent, n_heads, dropout, batch_first=True)
    
    def map_enc(self, map_obs: torch.Tensor) -> torch.Tensor:
        b, l, w, _ = map_obs.shape
        map_z = map_obs[..., 2].unsqueeze(dim=1)  # [b, 1, l, w]
        z_height = self.height_enc(map_z)  # [b, d-3, l, w]
        z_map = torch.cat((
            map_obs.view(b, 3, l, w),
            z_height
        ), dim=1)  # [b, d, l, w]
        return z_map.view(b, l * w, -1)  # [b, l * w, d]
    
    def forward(
        self,
        proprio_obs: torch.Tensor,
        map_obs: torch.Tensor,
        need_weights: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_map = self.map_enc(map_obs)  # [b, l * w, d]
        z_proprio = self.proprio_enc(proprio_obs).unsqueeze(dim=1) # [b, 1, d]
        z_mha, weights = self.mha(z_proprio, z_map, z_map, need_weights=need_weights)
        z = torch.cat((
            z_mha.squeeze(dim=1),
            proprio_obs,
        ), dim=-1)  # [b, d + d_obs]

        return z, (weights if need_weights else None)

def mlp(in_dim: int, out_dim: int, hidden: list[int], activation: str) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h)]
        layers += [resolve_nn_activation(activation)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class ActorCriticMHA(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        n_latent: int,
        n_heads: int,
        n_channels: int,
        kernel_size: int,
        n_rows: int,
        n_cols: int,
        dropout: float = 0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="relu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMHA.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        d_proprio = num_actor_obs - (n_rows * n_cols * 3)
        d_enc = n_latent + d_proprio
        self.encoder = MHAEncoder(
            d_proprio,
            n_latent,
            n_heads,
            n_channels,
            kernel_size,
            dropout,
            activation
        )

        self.actor_head = mlp(d_enc, num_actions, actor_hidden_dims, activation)
        self.critic_head = mlp(d_enc, 1, critic_hidden_dims, activation)
        self.l = n_rows
        self.w = n_cols

        print(f"MHA Encoder: {self.encoder}")
        print(f"Actor MLP: {self.actor_head}")
        print(f"Critic MLP: {self.critic_head}")

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
        # attention weight heatmap
        self.weights = None

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

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

    def get_attention_map(self, observations) -> torch.Tensor:
        proprio_obs = observations[:, :-self.w * self.l * 3]
        map_obs = observations[:, -self.w * self.l * 3:].view(-1, self.l, self.w, 3)
        _, weights = self.encoder(proprio_obs, map_obs, need_weights=True)
        return weights

    def update_distribution(self, observations, need_weights: bool = False):
        # compute mean
        proprio_obs = observations[:, :-self.w * self.l * 3]
        map_obs = observations[:, -self.w * self.l * 3:].view(-1, self.l, self.w, 3)
        z, self.weights = self.encoder(proprio_obs, map_obs, need_weights=False)
        mean = self.actor_head(z)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        proprio_obs = observations[:, :-self.w * self.l * 3]
        map_obs = observations[:, -self.w * self.l * 3:].view(-1, self.l, self.w, 3)
        z, _ = self.encoder(proprio_obs, map_obs, need_weights=False)
        actions_mean = self.actor_head(z)
        return actions_mean

    def act_inference_vis(self, observations):
        proprio_obs = observations[:, :-self.w * self.l * 3]
        map_obs = observations[:, -self.w * self.l * 3:].view(-1, self.l, self.w, 3)
        z, weights = self.encoder(proprio_obs, map_obs, need_weights=True)
        actions_mean = self.actor_head(z)
        return actions_mean, weights

    def evaluate(self, critic_observations, **kwargs):
        proprio_obs = critic_observations[:, :-self.w * self.l * 3]
        map_obs = critic_observations[:, -self.w * self.l * 3:].view(-1, self.l, self.w, 3)
        z, _ = self.encoder(proprio_obs, map_obs, need_weights=False)
        value = self.critic_head(z)
        return value

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
        return True

if __name__ == "__main__":
    num_actor_obs = 48
    num_critic_obs = 48
    num_actions = 12
    n_obs: int = 48
    n_latent: int = 64
    n_heads: int = 16
    n_channels: int = 16
    kernel_size: int = 5
    dropout: float = 0
    actor_hidden_dims=[256, 256, 256]
    critic_hidden_dims=[256, 256, 256]
    activation="relu"
    init_noise_std=1.0
    noise_std_type: str = "scalar"

    policy = ActorCriticMHA(
        num_actor_obs,
        num_critic_obs,
        num_actions,
        n_obs,
        n_latent,
        n_heads,
        n_channels,
        kernel_size,
        dropout,
        actor_hidden_dims,
        critic_hidden_dims,
        activation,
        init_noise_std,
        noise_std_type,
    )

    b = 4096
    proprio_obs = torch.randn(b, num_actor_obs)
    map_obs = torch.randn(b, 20, 20, 3)
    out = policy.act([proprio_obs, map_obs])
    weights = policy.get_attention_map([proprio_obs, map_obs])

    print(f"Proprio Obs: {proprio_obs.shape}")
    print(f"Map Obs: {map_obs.shape}")
    print(f"Actions: {out.shape}")
    print(f"Weights: {weights.shape}")