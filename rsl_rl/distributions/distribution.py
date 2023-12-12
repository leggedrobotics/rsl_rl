from abc import ABC, abstractmethod
import torch


class Distribution(ABC):
    def __init__(self, params: torch.Tensor) -> None:
        self._params = params

    @abstractmethod
    def sample(self, sample_count: int = 1) -> torch.Tensor:
        """Sample from the distribution.

        Args:
            sample_count: The number of samples to draw.
        Returns:
            A tensor of shape (sample_count, *parameter_shape).
        """
        pass
