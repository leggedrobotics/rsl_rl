from __future__ import annotations

import torch
import torch.nn as nn

from robot_rl.utils import resolve_nn_activation

class SAE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim_scale: int,
            sparsity_lambda: float = 1e-3,
            activation: str = "relu",
            **kwargs,
    ) -> None:
        if kwargs:
            print(
                "SAE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.activation = resolve_nn_activation(activation) if activation else nn.Identity()

        self.encoder = nn.Linear(input_dim, input_dim * hidden_dim_scale)
        self.decoder = nn.Linear(input_dim * hidden_dim_scale, input_dim)
        self.sparsity_lambda = sparsity_lambda

        print(f"SAE encoder: {self.encoder}")
        print(f"SAE decoder: {self.decoder}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.activation(self.encoder(x - self.decoder.bias))
        sparsity_loss = self.sparsity_lambda * torch.sum(torch.abs(encoded))
        decoded = self.decoder(encoded)
        return decoded, sparsity_loss