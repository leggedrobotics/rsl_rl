import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Union

from rsl_rl.modules.network import Network


class GaussianNetwork(Network):
    """A network to predict mean and std of a gaussian distribution where std is a tunable parameter."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        std_init: float = 1.0,
        **kwargs,
    ):
        super().__init__(input_size, output_size, **kwargs)

        self._log_std_max = log_std_max
        self._log_std_min = log_std_min

        self._log_std = nn.Parameter(torch.ones(output_size) * np.log(std_init))

    def forward(self, x: torch.Tensor, compute_std: bool = False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        mean = super().forward(x, **kwargs)

        if not compute_std:
            return mean

        log_std = torch.ones_like(mean) * self._log_std.clamp(self._log_std_min, self._log_std_max)
        std = log_std.exp()

        return mean, std
