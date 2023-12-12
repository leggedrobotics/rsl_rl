import torch
from typing import Any, Dict, Tuple, Union

from rsl_rl.algorithms.actor_critic import AbstractActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.storage.replay_storage import ReplayStorage
from rsl_rl.storage.storage import Dataset


class AbstractDPG(AbstractActorCritic):
    def __init__(
        self, env: VecEnv, action_noise_scale: float = 0.1, storage_initial_size=0, storage_size=100000, **kwargs
    ):
        """
        Args:
            env (VecEnv): A vectorized environment.
            action_noise_scale (float): The scale of the gaussian action noise.
            storage_initial_size (int): Initial size of the replay storage.
            storage_size (int): Maximum size of the replay storage.
        """
        assert action_noise_scale > 0

        super().__init__(env, **kwargs)

        self.storage = ReplayStorage(
            self.env.num_envs, storage_size, device=self.device, initial_size=storage_initial_size
        )

        self._register_serializable("storage")

        self._action_noise_scale = action_noise_scale

        self._register_serializable("_action_noise_scale")

    def draw_actions(
        self, obs: torch.Tensor, env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
        actor_obs, critic_obs = self._process_observations(obs, env_info)

        actions = self.actor.forward(actor_obs)
        noise = torch.normal(torch.zeros_like(actions), torch.ones_like(actions) * self._action_noise_scale)
        noisy_actions = self._process_actions(actions + noise)

        data = {"actor_observations": actor_obs.clone(), "critic_observations": critic_obs.clone()}

        return noisy_actions, data

    def register_terminations(self, terminations: torch.Tensor) -> None:
        pass
