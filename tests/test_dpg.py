import torch
import unittest
from rsl_rl.algorithms.dpg import AbstractDPG
from rsl_rl.env.pole_balancing import PoleBalancing


class DPG(AbstractDPG):
    def draw_actions(self, obs, env_info):
        pass

    def eval_mode(self):
        pass

    def get_inference_policy(self, device=None):
        pass

    def process_transition(
        self, observations, environment_info, actions, rewards, next_observations, next_environment_info, dones, data
    ):
        pass

    def register_terminations(self, terminations):
        pass

    def to(self, device):
        pass

    def train_mode(self):
        pass

    def update(self, storage):
        pass


class FakeCritic(torch.nn.Module):
    def __init__(self, values):
        self.values = values

    def forward(self, _):
        return self.values


class DPGTest(unittest.TestCase):
    def test_timeout_bootstrapping(self):
        env = PoleBalancing(environment_count=4)
        dpg = DPG(env, device="cpu", return_steps=3)

        rewards = torch.tensor(
            [
                [0.1000, 0.4000, 0.6000, 0.2000, -0.6000, -0.2000],
                [0.0000, 0.9000, 0.5000, -0.9000, -0.4000, 0.8000],
                [-0.5000, 0.4000, 0.0000, -0.2000, 0.3000, 0.1000],
                [-0.8000, 0.9000, -0.6000, 0.7000, 0.5000, 0.1000],
            ]
        )
        dones = torch.tensor(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ]
        )
        timeouts = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        )
        actions = torch.zeros((4, 6, 1))
        observations = torch.zeros((4, 6, 2))
        values = torch.tensor([-0.1000, -0.8000, 0.4000, 0.7000])

        dpg.critic = FakeCritic(values)
        dataset = [
            {
                "actions": actions[:, i],
                "critic_observations": observations[:, i],
                "dones": dones[:, i],
                "rewards": rewards[:, i],
                "timeouts": timeouts[:, i],
            }
            for i in range(3)
        ]

        processed_dataset = dpg._process_dataset(dataset)
        processed_rewards = torch.stack([processed_dataset[i]["rewards"] for i in range(1)], dim=-1)

        expected_rewards = torch.tensor(
            [
                [1.08406],
                [1.38105],
                [-0.5],
                [0.77707],
            ]
        )
        self.assertTrue(len(processed_dataset) == 1)
        self.assertTrue(torch.isclose(processed_rewards, expected_rewards).all())

        dataset = [
            {
                "actions": actions[:, i + 3],
                "critic_observations": observations[:, i + 3],
                "dones": dones[:, i + 3],
                "rewards": rewards[:, i + 3],
                "timeouts": timeouts[:, i + 3],
            }
            for i in range(3)
        ]

        processed_dataset = dpg._process_dataset(dataset)
        processed_rewards = torch.stack([processed_dataset[i]["rewards"] for i in range(3)], dim=-1)

        expected_rewards = torch.tensor(
            [
                [0.994, 0.6, -0.59002],
                [0.51291, -1.5592792, -2.08008],
                [0.20398, 0.09603, 0.19501],
                [1.593, 0.093, 0.7],
            ]
        )

        self.assertTrue(len(processed_dataset) == 3)
        self.assertTrue(torch.isclose(processed_rewards, expected_rewards).all())
