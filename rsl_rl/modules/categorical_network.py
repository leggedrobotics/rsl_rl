import torch
import torch.nn as nn
from rsl_rl.modules.network import Network
from rsl_rl.utils.utils import squeeze_preserve_batch

eps = torch.finfo(torch.float32).eps


class CategoricalNetwork(Network):
    def __init__(
        self,
        input_size,
        output_size,
        activations=["relu", "relu", "relu"],
        atom_count=51,
        hidden_dims=[256, 256, 256],
        init_gain=1.0,
        value_max=10.0,
        value_min=-10.0,
        **kwargs,
    ):
        assert len(hidden_dims) == len(activations)
        assert value_max > value_min
        assert atom_count > 1

        super().__init__(
            input_size,
            activations=activations,
            hidden_dims=hidden_dims[:-1],
            init_fade=False,
            init_gain=init_gain,
            output_size=hidden_dims[-1],
            **kwargs,
        )

        self._value_max = value_max
        self._value_min = value_min
        self._atom_count = atom_count

        self.value_delta = (self._value_max - self._value_min) / (self._atom_count - 1)
        action_values = torch.arange(self._value_min, self._value_max + eps, self.value_delta)
        self.register_buffer("action_values", action_values)

        self._categorical_layers = nn.ModuleList([nn.Linear(hidden_dims[-1], atom_count) for _ in range(output_size)])

        self._init(self._categorical_layers, fade=False, gain=init_gain)

    def categorical_loss(
        self, predictions: torch.Tensor, target_probabilities: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Computes the categorical loss between the prediction and target categorical distributions.

        Projects the targets back onto the categorical distribution supports before computing KL divergence.

        Args:
            predictions (torch.Tensor): The network prediction.
            target_probabilities (torch.Tensor): The next-state value probabilities.
            targets (torch.Tensor): The targets to compute the loss from.
        Returns:
            A torch.Tensor of the cross-entropy loss between the projected targets and the prediction.
        """
        b = (targets - self._value_min) / self.value_delta
        l = b.floor().long().clamp(0, self._atom_count - 1)
        u = b.ceil().long().clamp(0, self._atom_count - 1)

        all_idx = torch.arange(b.shape[0])
        projected_targets = torch.zeros((b.shape[0], self._atom_count), device=self.device)
        for i in range(self._atom_count):
            # Correct for when l == u
            l[:, i][(l[:, i] == u[:, i]) * (l[:, i] > 0)] -= 1
            u[:, i][(l[:, i] == u[:, i]) * (u[:, i] < self._atom_count - 1)] += 1

            projected_targets[all_idx, l[:, i]] += (u[:, i] - b[:, i]) * target_probabilities[..., i]
            projected_targets[all_idx, u[:, i]] += (b[:, i] - l[:, i]) * target_probabilities[..., i]

        loss = torch.nn.functional.cross_entropy(
            predictions.reshape(*projected_targets.shape), projected_targets.detach()
        )

        return loss

    def compute_targets(self, rewards: torch.Tensor, dones: torch.Tensor, discount: float) -> torch.Tensor:
        gamma = (discount * (1 - dones)).reshape(-1, 1)
        gamma_z = gamma * self.action_values.repeat(dones.size()[0], 1)
        targets = (rewards.reshape(-1, 1) + gamma_z).clamp(self._value_min, self._value_max)

        return targets

    def forward(self, x: torch.Tensor, distribution: bool = False) -> torch.Tensor:
        features = super().forward(x)
        probabilities = squeeze_preserve_batch(
            torch.stack([layer(features).softmax(dim=-1) for layer in self._categorical_layers], dim=1)
        )

        if distribution:
            return probabilities

        values = self.probabilities_to_values(probabilities)

        return values

    def probabilities_to_values(self, probabilities: torch.Tensor) -> torch.Tensor:
        values = probabilities @ self.action_values

        return values
