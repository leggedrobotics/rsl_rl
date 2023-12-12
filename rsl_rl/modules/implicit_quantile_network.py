import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
from typing import List, Union

from rsl_rl.modules.network import Network
from rsl_rl.modules.quantile_network import energy_loss
from rsl_rl.utils.benchmarkable import Benchmarkable


def reshape_measure_param(tau: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(param):
        param = torch.tensor([param])

    param = param.expand(tau.shape[0], -1).to(tau.device)

    return param


def risk_measure_neutral(tau: torch.Tensor) -> torch.Tensor:
    return tau


def risk_measure_wang(tau: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    beta = reshape_measure_param(tau, beta)

    distorted_tau = Normal(0, 1).cdf(Normal(0, 1).icdf(tau) + beta)

    return distorted_tau


class ImplicitQuantileNetwork(Network):
    measure_neutral = "neutral"
    measure_wang = "wang"

    measures = {
        measure_neutral: risk_measure_neutral,
        measure_wang: risk_measure_wang,
    }

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activations: List[str] = ["relu", "relu", "relu"],
        feature_layers: int = 1,
        embedding_size: int = 64,
        hidden_dims: List[int] = [256, 256, 256],
        init_fade: bool = False,
        init_gain: float = 0.5,
        measure: str = None,
        measure_kwargs: dict = {},
        **kwargs,
    ):
        assert len(hidden_dims) == len(activations), "hidden_dims and activations must have the same length."
        assert feature_layers > 0, "feature_layers must be greater than 0."
        assert feature_layers < len(hidden_dims), "feature_layers must be less than the number of hidden dimensions."
        assert embedding_size > 0, "embedding_size must be greater than 0."

        super().__init__(
            input_size,
            hidden_dims[feature_layers - 1],
            activations=activations[:feature_layers],
            hidden_dims=hidden_dims[: feature_layers - 1],
            init_fade=init_fade,
            init_gain=init_gain,
            **kwargs,
        )

        self._last_taus = None
        self._last_quantiles = None
        self._embedding_size = embedding_size
        self.register_buffer(
            "_embedding_pis",
            np.pi * (torch.arange(self._embedding_size, device=self.device).reshape(1, 1, self._embedding_size)),
        )
        self._embedding_layer = nn.Sequential(
            nn.Linear(self._embedding_size, hidden_dims[feature_layers - 1]), nn.ReLU()
        )

        self._fusion_layers = Network(
            hidden_dims[feature_layers - 1],
            output_size,
            activations=activations[feature_layers:] + ["linear"],
            hidden_dims=hidden_dims[feature_layers:],
            init_fade=init_fade,
            init_gain=init_gain,
        )

        measure_func = risk_measure_neutral if measure is None else self.measures[measure]
        self._measure_func = measure_func
        self._measure_kwargs = measure_kwargs

    @Benchmarkable.register
    def _sample_taus(self, batch_size: int, sample_count: int, measure_args: list, use_measure: bool) -> torch.Tensor:
        """Sample quantiles and distort them according to the risk metric.

        Args:
            batch_size: The batch size.
            sample_count: The number of samples to draw.
            measure_args: The arguments to pass to the risk measure function.
            use_measure: Whether to use the risk measure or not.
        Returns:
            A tensor of shape (batch_size, sample_count, 1).
        """
        taus = torch.rand(batch_size, sample_count, device=self.device)

        if not use_measure:
            return taus

        if measure_args:
            taus = self._measure_func(taus, *measure_args)
        else:
            taus = self._measure_func(taus, **self._measure_kwargs)

        return taus

    @Benchmarkable.register
    def forward(
        self,
        x: torch.Tensor,
        distribution: bool = False,
        measure_args: list = [],
        sample_count: int = 8,
        taus: Union[torch.Tensor, None] = None,
        use_measure: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        assert taus is None or not use_measure, "Cannot use taus and use_measure at the same time."

        batch_size = x.shape[0]

        features = super().forward(x, **kwargs)
        taus = self._sample_taus(batch_size, sample_count, measure_args, use_measure) if taus is None else taus

        # Compute quantile embeddings
        singular_dims = [1] * taus.dim()
        cos = torch.cos(taus.unsqueeze(-1) * self._embedding_pis.reshape(*singular_dims, self._embedding_size))
        embeddings = self._embedding_layer(cos)

        # Compute the fusion of the features and the embeddings
        fused_features = features.unsqueeze(-2) * embeddings
        quantiles = self._fusion_layers(fused_features)

        self._last_quantiles = quantiles
        self._last_taus = taus

        if distribution:
            return quantiles

        values = quantiles.mean(-1)

        return values

    @property
    def last_taus(self):
        return self._last_taus

    @property
    def last_quantiles(self):
        return self._last_quantiles

    @Benchmarkable.register
    def sample_energy_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = energy_loss(predictions, targets)

        return loss
