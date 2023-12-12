import unittest

from rsl_rl.algorithms import D4PG, DDPG, DPPO, DSAC, PPO, SAC, TD3
from rsl_rl.env.gym_env import GymEnv
from rsl_rl.modules import Network
from rsl_rl.runners.runner import Runner

DEVICE = "cpu"


class AlgorithmTestCaseMixin:
    algorithm_class = None

    def _make_env(self, params={}):
        my_params = dict(name="LunarLanderContinuous-v2", device=DEVICE, environment_count=4)
        my_params.update(params)

        return GymEnv(**my_params)

    def _make_agent(self, env, agent_params={}):
        return self.algorithm_class(env, device=DEVICE, **agent_params)

    def _make_runner(self, env, agent, runner_params={}):
        if not runner_params or "num_steps_per_env" not in runner_params:
            runner_params["num_steps_per_env"] = 6

        return Runner(env, agent, device=DEVICE, **runner_params)

    def _learn(self, env, agent, runner_params={}):
        runner = self._make_runner(env, agent, runner_params)
        runner.learn(10)

    def test_default(self):
        env = self._make_env()
        agent = self._make_agent(env)

        self._learn(env, agent)

    def test_single_env_single_step(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env)

        self._learn(env, agent, dict(num_steps_per_env=1))


class RecurrentAlgorithmTestCaseMixin(AlgorithmTestCaseMixin):
    def test_recurrent(self):
        env = self._make_env()
        agent = self._make_agent(env, dict(recurrent=True))

        self._learn(env, agent)

    def test_single_env_single_step_recurrent(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env, dict(recurrent=True))

        self._learn(env, agent, dict(num_steps_per_env=1))


class D4PGTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = D4PG


class DDPGTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = DDPG


iqn_params = dict(
    critic_network=DPPO.network_iqn,
    iqn_action_samples=8,
    iqn_embedding_size=16,
    iqn_feature_layers=2,
    iqn_value_samples=4,
    value_loss=DPPO.value_loss_energy,
)

qrdqn_params = dict(
    critic_network=DPPO.network_qrdqn,
    qrdqn_quantile_count=16,
    value_loss=DPPO.value_loss_l1,
)


class DPPOTest(RecurrentAlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = DPPO

    def test_qrdqn(self):
        env = self._make_env()
        agent = self._make_agent(env, qrdqn_params)

        self._learn(env, agent)

    def test_qrdqn_sing_env_single_step(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env, qrdqn_params)

        self._learn(env, agent, dict(num_steps_per_env=1))

    def test_qrdqn_energy_loss(self):
        my_agent_params = qrdqn_params.copy()
        my_agent_params["value_loss"] = DPPO.value_loss_energy

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)

    def test_qrdqn_huber_loss(self):
        my_agent_params = qrdqn_params.copy()
        my_agent_params["value_loss"] = DPPO.value_loss_huber

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)

    def test_qrdqn_transformer(self):
        my_agent_params = qrdqn_params.copy()
        my_agent_params["recurrent"] = True
        my_agent_params["critic_recurrent_layers"] = 2
        my_agent_params["critic_recurrent_module"] = Network.recurrent_module_transformer
        my_agent_params["critic_recurrent_tf_context_length"] = 8
        my_agent_params["critic_recurrent_tf_head_count"] = 2

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)

    def test_iqn(self):
        env = self._make_env()
        agent = self._make_agent(env, iqn_params)

        self._learn(env, agent)

    def test_iqn_single_step_single_env(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env, iqn_params)

        self._learn(env, agent, dict(num_steps_per_env=1))

    def test_iqn_recurrent(self):
        my_agent_params = iqn_params.copy()
        my_agent_params["recurrent"] = True

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)


class DSACTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = DSAC


class PPOTest(RecurrentAlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = PPO


class SACTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = SAC


class TD3Test(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = TD3


if __name__ == "__main__":
    unittest.main()
