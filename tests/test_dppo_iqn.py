import torch
import unittest
from rsl_rl.algorithms import DPPO
from rsl_rl.env.vec_env import VecEnv

ACTION_SIZE = 3
ENV_COUNT = 3
OBS_SIZE = 24


class FakeEnv(VecEnv):
    def __init__(self, rewards, dones, environment_count=1):
        super().__init__(OBS_SIZE, OBS_SIZE, environment_count=environment_count)

        self.num_actions = ACTION_SIZE
        self.rewards = rewards
        self.dones = dones

        self._step = 0

    def get_observations(self):
        return torch.zeros((self.num_envs, self.num_obs)), {"observations": {}}

    def get_privileged_observations(self):
        return torch.zeros((self.num_envs, self.num_privileged_obs)), {"observations": {}}

    def step(self, actions):
        obs, _ = self.get_observations()
        rewards = self.rewards[self._step]
        dones = self.dones[self._step]

        self._step += 1

        return obs, rewards, dones, {"observations": {}}

    def reset(self):
        pass


class FakeCritic(torch.nn.Module):
    def __init__(self, action_samples, value_samples, action_values, value_values, action_taus, value_taus):
        self.recurrent = False
        self.action_samples = action_samples
        self.value_samples = value_samples
        self.action_values = action_values
        self.value_values = value_values
        self.action_taus = action_taus
        self.value_taus = value_taus

        self.last_quantiles = None
        self.last_taus = None

    def forward(self, _, distribution=False, measure_args=None, sample_count=8, taus=None, use_measure=True):
        if taus is not None:
            sample_count = taus.shape[-1]

        if sample_count == self.action_samples:
            self.last_taus = self.action_taus
            self.last_quantiles = self.action_values
        elif sample_count == self.value_samples:
            self.last_taus = self.value_taus
            self.last_quantiles = self.value_values
        else:
            raise ValueError(f"Invalid sample count: {sample_count}")

        if distribution:
            return self.last_quantiles

        return self.last_quantiles.mean(-1)


def fake_process_quants(self, x):
    idx = torch.arange(0, x.shape[-1]).expand(*x.shape[:-1])

    return x, idx


class DPPOTest(unittest.TestCase):
    def test_value_target_computation(self):
        rewards = torch.tensor(
            [
                [-1.0000e02, -1.4055e-01, -3.0476e-02],
                [-1.7633e-01, -2.6533e-01, -3.0786e-01],
                [-1.5952e-01, -1.5177e-01, -1.4296e-01],
                [1.1407e-02, -1.0000e02, -6.2290e-02],
            ]
        )
        dones = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
            ]
        )

        env = FakeEnv(rewards, dones, environment_count=ENV_COUNT)
        dppo = DPPO(
            env,
            critic_network=DPPO.network_iqn,
            device="cpu",
            gae_lambda=0.97,
            gamma=0.99,
            iqn_action_samples=4,
            iqn_value_samples=2,
            value_lambda=1.0,
            value_loss=DPPO.value_loss_energy,
        )

        # Generate fake dataset

        action_taus = torch.tensor(
            [
                [[0.3, 0.5, 1.0, 0.2], [0.8, 0.9, 0.0, 0.9], [0.6, 0.1, 0.6, 0.5]],
                [[0.7, 0.9, 0.3, 0.0], [1.0, 0.7, 0.7, 0.7], [0.3, 0.8, 0.8, 0.1]],
                [[0.3, 0.8, 0.3, 0.2], [0.2, 0.9, 0.6, 0.4], [0.8, 0.4, 0.8, 1.0]],
                [[0.6, 0.6, 0.8, 0.8], [0.8, 0.0, 0.9, 0.1], [0.2, 0.3, 0.6, 0.2]],
            ]
        )
        action_value_quants = torch.tensor(
            [
                [[0.2, 0.2, 0.6, 0.5], [0.5, 0.8, 0.1, 0.0], [1.0, 0.1, 0.8, 0.8]],
                [[0.0, 0.6, 0.1, 0.9], [0.2, 1.0, 0.9, 1.0], [0.4, 0.1, 0.1, 0.8]],
                [[0.7, 0.0, 0.6, 0.8], [0.7, 0.7, 0.7, 0.8], [0.0, 0.1, 0.5, 0.8]],
                [[0.5, 0.8, 0.1, 0.1], [0.9, 0.4, 0.7, 0.6], [0.6, 0.3, 0.1, 0.4]],
            ]
        )
        value_taus = torch.tensor(
            [
                [[0.3, 0.5], [0.8, 0.9], [0.6, 0.1]],
                [[0.7, 0.9], [1.0, 0.7], [0.3, 0.8]],
                [[0.3, 0.8], [0.2, 0.9], [0.8, 0.4]],
                [[0.6, 0.6], [0.8, 0.0], [0.2, 0.3]],
            ]
        )
        value_value_quants = torch.tensor(
            [
                [[0.9, 0.8], [0.1, 0.3], [0.3, 0.5]],
                [[0.2, 0.1], [0.9, 0.3], [0.4, 0.2]],
                [[0.7, 1.0], [0.6, 0.2], [0.2, 0.6]],
                [[0.4, 1.0], [0.3, 0.6], [0.3, 0.1]],
            ]
        )

        actions = torch.zeros(ENV_COUNT, ACTION_SIZE)
        env_info = {"observations": {}}
        obs = torch.zeros(ENV_COUNT, OBS_SIZE)
        dataset = []
        for i in range(4):
            dppo.critic = FakeCritic(4, 2, action_value_quants[i], value_value_quants[i], action_taus[i], value_taus[i])
            dppo.critic._process_quants = fake_process_quants

            _, data = dppo.draw_actions(obs, {})
            _, rewards, dones, _ = env.step(actions)

            dataset.append(
                dppo.process_transition(
                    obs,
                    env_info,
                    actions,
                    rewards,
                    obs,
                    env_info,
                    dones,
                    data,
                )
            )

        processed_dataset = dppo._process_dataset(dataset)

        # TODO: Test that the value targets are correct.
