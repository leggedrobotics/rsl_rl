from abc import abstractmethod
import torch
from typing import Dict, Generator, List

from rsl_rl.utils.serializable import Serializable


# prev_obs, prev_obs_info, actions, rewards, next_obs, next_obs_info, dones, data
Transition = Dict[str, torch.Tensor]
Dataset = List[Transition]


class Storage(Serializable):
    @abstractmethod
    def append(self, dataset: Dataset) -> None:
        """Adds transitions to the storage.

        Args:
            dataset (Dataset): The transitions to add to the storage.
        """
        pass

    @abstractmethod
    def batch_generator(self, batch_size: int, batch_count: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generates a batch of transitions.

        Args:
            batch_size (int): The size of each batch to generate.
            batch_count (int): The number of batches to generate.
        Returns:
            A generator that yields transitions.
        """
        pass

    @property
    def initialized(self) -> bool:
        """Returns whether the storage is initialized."""
        return True

    @abstractmethod
    def sample_count(self) -> int:
        """Returns how many individual samples are stored in the storage.

        Returns:
            The number of stored samples.
        """
        pass
