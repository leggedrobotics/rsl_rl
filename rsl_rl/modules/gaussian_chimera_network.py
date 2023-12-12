import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Union

from rsl_rl.modules.network import Network
from rsl_rl.modules.utils import get_activation


class GaussianChimeraNetwork(Network):
    """A network to predict mean and std of a gaussian distribution with separate heads for mean and std."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activations: List[str] = ["relu", "relu", "relu", "linear"],
        hidden_dims: List[int] = [256, 256, 256],
        init_fade: bool = True,
        init_gain: float = 0.5,
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        std_init: float = 1.0,
        shared_dims: int = 1,
        **kwargs,
    ):
        assert len(hidden_dims) + 1 == len(activations)
        assert shared_dims > 0 and shared_dims <= len(hidden_dims)

        super().__init__(
            input_size,
            hidden_dims[shared_dims],
            activations=activations[: shared_dims + 1],
            hidden_dims=hidden_dims[:shared_dims],
            init_fade=False,
            init_gain=init_gain,
            **kwargs,
        )

        # Since the network predicts log_std ~= 0 after initialization, compute std = std_init * exp(log_std).
        self._log_std_init = np.log(std_init)
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min

        separate_dims = len(hidden_dims) - shared_dims

        mean_layers = []
        for i in range(separate_dims):
            isize = hidden_dims[shared_dims + i]
            osize = output_size if i == separate_dims - 1 else hidden_dims[shared_dims + i + 1]

            layer = nn.Linear(isize, osize)
            activation = activations[shared_dims + i + 1]

            mean_layers += [layer, get_activation(activation)]
        self._mean_layer = nn.Sequential(*mean_layers)

        self._init(self._mean_layer, fade=init_fade, gain=init_gain)

        log_std_layers = []
        for i in range(separate_dims):
            isize = hidden_dims[shared_dims + i]
            osize = output_size if i == separate_dims - 1 else hidden_dims[shared_dims + i + 1]

            layer = nn.Linear(isize, osize)
            activation = activations[shared_dims + i + 1]

            log_std_layers += [layer, get_activation(activation)]
        self._log_std_layer = nn.Sequential(*log_std_layers)

        self._init(self._log_std_layer, fade=init_fade, gain=init_gain)

    def forward(self, x: torch.Tensor, compute_std: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        features = super().forward(x)

        mean = self._mean_layer(features)

        if not compute_std:
            return mean

        # compute standard deviation as std = std_init * exp(log_std) = exp(log(std_init) + log(std)) since the network
        # will predict log_std ~= 0 after initialization.
        log_std = (self._log_std_init + self._log_std_layer(features)).clamp(self._log_std_min, self._log_std_max)
        std = log_std.exp()

        return mean, std
