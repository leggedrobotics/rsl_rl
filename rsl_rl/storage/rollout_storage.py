import torch
from typing import Generator

from rsl_rl.storage.replay_storage import ReplayStorage
from rsl_rl.storage.storage import Dataset, Transition


class RolloutStorage(ReplayStorage):
    """Implementation of rollout storage for RL-agent."""

    def __init__(self, environment_count: int, device: str = "cpu"):
        """
        Args:
            environment_count (int): Number of environments.
            device (str, optional): Device to use. Defaults to "cpu".
        """
        super().__init__(environment_count, environment_count, device=device, initial_size=0)

        self._size_initialized = False

    def append(self, dataset: Dataset) -> None:
        """Appends a dataset to the rollout storage.

        Args:
            dataset (Dataset): Dataset to append.
        Raises:
            AssertionError: If the dataset is not of the correct size.
        """
        assert self._idx == 0

        if not self._size_initialized:
            self.max_size = len(dataset) * self._env_count

        assert len(dataset) == self._size

        super().append(dataset)

    def batch_generator(self, batch_count: int, trajectories: bool = False) -> Generator[Transition, None, None]:
        """Yields batches of transitions or trajectories.

        Args:
            batch_count (int): Number of batches to yield.
            trajectories (bool, optional): Whether to yield batches of trajectories. Defaults to False.
        Raises:
            AssertionError: If the rollout storage is not full.
        Returns:
            Generator yielding batches of transitions of shape (batch_size, *shape). If trajectories is True, yields
            batches of trajectories of shape (env_count, steps_per_env, *shape).
        """
        assert self._full and self._initialized, "Rollout storage must be full and initialized."

        total_size = self._env_count if trajectories else self._size * self._env_count
        batch_size = total_size // batch_count
        indices = torch.randperm(total_size)

        assert batch_size > 0, "Batch count is too large."

        if trajectories:
            # Reshape to (env_count, steps_per_env, *shape)
            data = {k: v.reshape(-1, self._env_count, *v.shape[1:]).transpose(0, 1) for k, v in self._data.items()}
        else:
            data = self._data

        for i in range(batch_count):
            batch_idx = indices[i * batch_size : (i + 1) * batch_size].detach().to(self.device)

            batch = {}
            for key, value in data.items():
                batch[key] = self._process_undo(key, value[batch_idx].clone())

            yield batch
