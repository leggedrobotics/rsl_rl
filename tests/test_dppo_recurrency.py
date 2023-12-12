import torch
import unittest
from rsl_rl.algorithms import DPPO
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners.runner import Runner
from rsl_rl.utils.benchmarkable import Benchmarkable


class FakeNetwork(torch.nn.Module, Benchmarkable):
    def __init__(self, values):
        super().__init__()

        self.hidden_state = None
        self.quantile_count = 1
        self.recurrent = True
        self.values = values

        self._hidden_size = 2

    def forward(self, x, hidden_state=None):
        if not hidden_state:
            self.hidden_state = (self.hidden_state[0] + 1, self.hidden_state[1] - 1)

        values = self.values.repeat((*x.shape[:-1], 1)).squeeze(-1)
        values.requires_grad_(True)

        return values

    def reset_full_hidden_state(self, batch_size=None):
        assert batch_size is None or batch_size == 4, f"batch_size={batch_size}"

        self.hidden_state = (torch.zeros((1, 4, self._hidden_size)), torch.zeros((1, 4, self._hidden_size)))

    def reset_hidden_state(self, indices):
        self.hidden_state[0][:, indices] = torch.zeros((len(indices), self._hidden_size))
        self.hidden_state[1][:, indices] = torch.zeros((len(indices), self._hidden_size))


class FakeActorNetwork(FakeNetwork):
    def forward(self, x, compute_std=False, hidden_state=None):
        values = super().forward(x, hidden_state=hidden_state)

        if compute_std:
            return values, torch.ones_like(values)

        return values


class FakeCriticNetwork(FakeNetwork):
    _quantile_count = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, distribution=False, hidden_state=None, measure_args=None):
        values = super().forward(x, hidden_state=hidden_state)
        self.last_quantiles = values.reshape(*values.shape, 1)

        if distribution:
            return self.last_quantiles

        return values

    def quantile_l1_loss(self, *args, **kwargs):
        return torch.tensor(0.0)

    def quantiles_to_values(self, quantiles):
        return quantiles.squeeze()


class FakeEnv(VecEnv):
    def __init__(self, dones=None, **kwargs):
        super().__init__(3, 3, **kwargs)

        self.num_actions = 3
        self._extra = {"observations": {}, "time_outs": torch.zeros((self.num_envs, 1))}

        self._step = 0
        self._dones = dones

        self.reset()

    def get_observations(self):
        return self._state_buf, self._extra

    def get_privileged_observations(self):
        return self._state_buf, self._extra

    def reset(self):
        self._state_buf = torch.zeros((self.num_envs, self.num_obs))

        return self._state_buf, self._extra

    def step(self, actions):
        assert actions.shape[0] == self.num_envs
        assert actions.shape[1] == self.num_actions

        self._state_buf += actions

        rewards = torch.zeros((self.num_envs))
        dones = torch.zeros((self.num_envs)) if self._dones is None else self._dones[self._step % self._dones.shape[0]]

        self._step += 1

        return self._state_buf, rewards, dones, self._extra


class DPPORecurrencyTest(unittest.TestCase):
    def test_draw_action_produces_hidden_state(self):
        """Test that the hidden state is correctly added to the data dictionary when drawing actions."""
        env = FakeEnv(environment_count=4)
        dppo = DPPO(env, device="cpu", recurrent=True)

        dppo.actor = FakeActorNetwork(torch.ones(env.num_actions))
        dppo.critic = FakeCriticNetwork(torch.zeros(1))

        # Done during DPPO.__init__, however we need to reset the hidden state here again since we are using a fake
        # network that is added after initialization.
        dppo.actor.reset_full_hidden_state(batch_size=env.num_envs)
        dppo.critic.reset_full_hidden_state(batch_size=env.num_envs)

        ones = torch.ones((1, env.num_envs, dppo.actor._hidden_size))
        state, extra = env.reset()
        for ctr in range(10):
            _, data = dppo.draw_actions(state, extra)

            # Actor state is changed every time an action is drawn.
            self.assertTrue(torch.allclose(data["actor_state_h"], ones * ctr))
            self.assertTrue(torch.allclose(data["actor_state_c"], -ones * ctr))
            # Critic state is only changed and saved when processing the transition (evaluating the action) so we can't
            # check it here.

    def test_update_produces_hidden_state(self):
        """Test that the hidden state is correctly added to the data dictionary when updating."""
        dones = torch.cat((torch.tensor([[0, 0, 0, 1]]), torch.zeros((4, 4)), torch.tensor([[1, 0, 0, 0]])), dim=0)

        env = FakeEnv(dones=dones, environment_count=4)
        dppo = DPPO(env, device="cpu", recurrent=True)
        runner = Runner(env, dppo, num_steps_per_env=6)

        dppo._value_loss = lambda *args, **kwargs: torch.tensor(0.0)

        dppo.actor = FakeActorNetwork(torch.ones(env.num_actions))
        dppo.critic = FakeCriticNetwork(torch.zeros(1))

        dppo.actor.reset_full_hidden_state(batch_size=env.num_envs)
        dppo.critic.reset_full_hidden_state(batch_size=env.num_envs)

        runner.learn(1)

        state_h_0 = torch.tensor([[0, 0], [0, 0], [0, 0], [0, 0]])
        state_h_1 = torch.tensor([[1, 1], [1, 1], [1, 1], [0, 0]])
        state_h_2 = state_h_1 + 1
        state_h_3 = state_h_2 + 1
        state_h_4 = state_h_3 + 1
        state_h_5 = state_h_4 + 1
        state_h_6 = torch.tensor([[0, 0], [6, 6], [6, 6], [5, 5]])
        state_h = (
            torch.cat((state_h_0, state_h_1, state_h_2, state_h_3, state_h_4, state_h_5), dim=0).float().unsqueeze(1)
        )
        next_state_h = (
            torch.cat((state_h_1, state_h_2, state_h_3, state_h_4, state_h_5, state_h_6), dim=0).float().unsqueeze(1)
        )

        self.assertTrue(torch.allclose(dppo.storage._data["critic_state_h"], state_h))
        self.assertTrue(torch.allclose(dppo.storage._data["critic_state_c"], -state_h))
        self.assertTrue(torch.allclose(dppo.storage._data["critic_next_state_h"], next_state_h))
        self.assertTrue(torch.allclose(dppo.storage._data["critic_next_state_c"], -next_state_h))
        self.assertTrue(torch.allclose(dppo.storage._data["actor_state_h"], state_h))
        self.assertTrue(torch.allclose(dppo.storage._data["actor_state_c"], -state_h))
