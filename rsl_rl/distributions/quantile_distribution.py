import torch

from .distribution import Distribution


class QuantileDistribution(Distribution):
    def sample(self, sample_count: int = 1) -> torch.Tensor:
        idx = torch.randint(
            self._params.shape[-1], (*self._params.shape[:-1], sample_count), device=self._params.device
        )
        samples = torch.take_along_dim(self._params, idx, -1)

        return samples, idx
