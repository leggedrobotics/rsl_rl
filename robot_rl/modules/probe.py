from __future__ import annotations

import torch
import torch.nn as nn

from robot_rl.utils import resolve_nn_activation

class Probe(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            probe_hidden_dims: list = [],
            activation: str = "elu",
            **kwargs,
    ) -> None:
        if kwargs:
            print(
                "Probe.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if probe_hidden_dims:
            probe_layers = []
            probe_layers.append(nn.Linear(input_dim, probe_hidden_dims[0]))
            probe_layers.append(resolve_nn_activation(activation))
            for layer_index in range(len(probe_hidden_dims)):
                if layer_index == len(probe_hidden_dims) - 1:
                    probe_layers.append(nn.Linear(probe_hidden_dims[layer_index], output_dim))
                else:
                    probe_layers.append(nn.Linear(probe_hidden_dims[layer_index], probe_hidden_dims[layer_index + 1]))
                    probe_layers.append(resolve_nn_activation(activation))
            self.probe = nn.Sequential(*probe_layers)

        else:
            self.probe = nn.Sequential(nn.Linear(input_dim, output_dim))

        print(f"Probe MLP: {self.probe}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales) -> None:
        [
            nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self, z) -> torch.Tensor:
        return self.probe(z)