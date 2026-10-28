# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PPO algorithm."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import CNNModel, MLPModel, RNNModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4
RND_CFG = {
    "num_states": OBS_DIM,
    "obs_groups": {"rnd_state": ["policy"]},
    "num_outputs": 4,
    "predictor_hidden_dims": [32],
    "target_hidden_dims": [32],
    "weight": 2.0,
    "weight_schedule": {"mode": "step", "final_step": 4, "final_value": 0.25},
}


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: object) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, object] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critic(obs: TensorDict, obs_groups: dict, **kwargs: object) -> MLPModel:
    """Create an MLPModel critic (no distribution)."""
    defaults: dict[str, object] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "critic", 1, **defaults)


def _build_ppo(**overrides: object) -> tuple[PPO, TensorDict]:
    """Build a PPO instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
    critic = _make_critic(obs, obs_groups)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        schedule="fixed",
        desired_kl=0.01,
    )
    defaults.update(overrides)
    ppo = PPO(actor, critic, storage, **defaults)
    return ppo, obs


def _fill_rollout(ppo: PPO, obs: TensorDict) -> None:
    """Collect transitions for one PPO update."""
    with torch.no_grad():
        for _ in range(ppo.storage.num_transitions_per_env):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})


def _build_ppo_with_shared_cnn() -> PPO:
    """Build a PPO instance whose actor and critic share a CNN encoder."""
    obs = TensorDict(
        {"policy": torch.randn(NUM_ENVS, OBS_DIM), "image": torch.randn(NUM_ENVS, 1, 8, 8)}, batch_size=[NUM_ENVS]
    )
    obs_groups = {"actor": ["policy", "image"], "critic": ["policy", "image"]}
    cnn_cfg = {"image": {"output_channels": [4], "kernel_size": 3, "stride": 1, "activation": "elu"}}
    actor = CNNModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32],
        cnn_cfg=cnn_cfg,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = CNNModel(obs, obs_groups, "critic", 1, hidden_dims=[32], cnns=actor.cnns)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    return PPO(actor, critic, storage, learning_rate=1e-3)


class TestGAEComputation:
    """Tests for generalized advantage estimation in ``compute_returns``."""

    @pytest.mark.parametrize("per_batch", [False, True])
    def test_singleton_advantage_is_preserved(self, per_batch: bool) -> None:
        """A singleton retains its learning signal through global and per-batch normalization."""
        obs = make_obs(1, OBS_DIM)
        groups = {"actor": ["policy"], "critic": ["policy"]}
        storage = RolloutStorage("rl", 1, 1, obs, [NUM_ACTIONS])
        ppo = PPO(
            _make_actor(obs, groups),
            _make_critic(obs, groups),
            storage,
            gamma=0.0,
            num_learning_epochs=1,
            num_mini_batches=1,
            schedule="fixed",
            normalize_advantage_per_mini_batch=per_batch,
        )
        with torch.no_grad():
            ppo.act(obs)
            reward = ppo.transition.values.flatten() + 2.0
            ppo.process_env_step(obs, reward, torch.zeros(1), {})
            ppo.compute_returns(obs)

        torch.testing.assert_close(storage.advantages, torch.full((1, 1, 1), 2.0))
        losses = ppo.update()
        assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        assert losses["surrogate"] == pytest.approx(-2.0)

    def test_gae_returns_hand_computed(self) -> None:
        """Verify GAE returns match a hand-computed example with known rewards, values, and dones."""
        num_envs, num_steps = 1, 3
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [0.0, 0.0, 0.0]

        for i in range(num_steps):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), values[i])
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), rewards[i])
            t.dones = torch.full((num_envs,), dones[i])
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 2.0)
        # Manually compute GAE (backward pass)
        # Step 2: delta = r2 + gamma * V_last - V2 = 3.0 + 0.99*2.0 - 1.5 = 3.48
        #          adv2 = 3.48
        # Step 1: delta = r1 + gamma * V2 - V1 = 2.0 + 0.99*1.5 - 1.0 = 2.485
        #          adv1 = 2.485 + gamma*lam*adv2 = 2.485 + 0.99*0.95*3.48 = 2.485 + 3.27294 = 5.75794
        # Step 0: delta = r0 + gamma * V1 - V0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        #          adv0 = 1.49 + gamma*lam*adv1 = 1.49 + 0.99*0.95*5.75794 = 1.49 + 5.41484... = 6.90484...
        expected_adv = [
            1.49 + 0.99 * 0.95 * (2.485 + 0.99 * 0.95 * 3.48),
            2.485 + 0.99 * 0.95 * 3.48,
            3.48,
        ]
        expected_returns = [expected_adv[i] + values[i] for i in range(3)]

        # Call compute_returns with a custom last_values by monkeypatching critic
        original_critic_call = ppo.critic.forward
        ppo.critic.forward = lambda *a, **kw: last_values
        ppo.compute_returns(obs)
        ppo.critic.forward = original_critic_call

        for i in range(num_steps):
            assert torch.allclose(
                storage.returns[i, 0, 0],
                torch.tensor(expected_returns[i]),
                atol=1e-4,
            ), f"Return mismatch at step {i}: got {storage.returns[i, 0, 0].item()}, expected {expected_returns[i]}"

    def test_gae_terminal_state_cuts_bootstrap(self) -> None:
        """When a done flag is set, the advantage should not bootstrap from the next value."""
        num_envs, num_steps = 1, 2
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critic = _make_critic(obs, obs_groups)
        storage = RolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS])
        ppo = PPO(
            actor, critic, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        # Step 0: done=True, so step 1 is a fresh episode
        for i, (r, v, d) in enumerate([(1.0, 0.5, 1.0), (2.0, 1.0, 0.0)]):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = torch.full((num_envs, 1), v)
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), r)
            t.dones = torch.full((num_envs,), d)
            storage.add_transition(t)

        last_values = torch.full((num_envs, 1), 3.0)
        ppo.critic.forward = lambda *a, **kw: last_values
        ppo.compute_returns(obs)

        # Step 0: done=True, so next_is_not_terminal = 0
        # delta0 = r0 - V0 = 1.0 - 0.5 = 0.5 (no bootstrap because done)
        # Step 1: delta1 = r1 + gamma * V_last - V1 = 2.0 + 0.99*3.0 - 1.0 = 3.97
        # adv1 = 3.97
        # adv0 = 0.5 (no bootstrap because done at step 0)
        expected_return_0 = 0.5 + 0.5  # adv0 + V0
        expected_return_1 = 3.97 + 1.0  # adv1 + V1

        assert torch.allclose(storage.returns[0, 0, 0], torch.tensor(expected_return_0), atol=1e-4)
        assert torch.allclose(storage.returns[1, 0, 0], torch.tensor(expected_return_1), atol=1e-4)

    def test_advantage_normalization_global(self) -> None:
        """With normalize_advantage_per_mini_batch=False, advantages should have mean~0, std~1."""
        ppo, obs = _build_ppo(normalize_advantage_per_mini_batch=False)

        for _ in range(NUM_STEPS):
            t = RolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None, None)
            t.actions = ppo.actor(obs, stochastic_output=True).detach()
            t.values = ppo.critic(obs).detach()
            t.actions_log_prob = ppo.actor.get_output_log_prob(t.actions).detach()
            t.distribution_params = tuple(p.detach() for p in ppo.actor.output_distribution_params)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            ppo.storage.add_transition(t)

        ppo.compute_returns(obs)

        adv = ppo.storage.advantages.flatten()
        assert abs(adv.mean().item()) < 1e-5, "Advantages should be zero-mean"
        assert abs(adv.std().item() - 1.0) < 0.1, "Advantages should be unit-std"


class TestTimeoutBootstrapping:
    """Tests for timeout bootstrapping in ``process_env_step``."""

    def test_timeout_adds_bootstrap_to_reward(self) -> None:
        """When time_outs is set, stored reward should include gamma * value * timeout."""
        ppo, obs = _build_ppo()

        # Manually act to populate transition.values
        ppo.act(obs)
        stored_values = ppo.transition.values.clone()

        raw_reward = torch.ones(NUM_ENVS)
        dones = torch.ones(NUM_ENVS)
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0  # Only env 0 times out

        ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})

        # The stored reward for env 0 should be: 1.0 + gamma * value[0]
        stored_reward_env0 = ppo.storage.rewards[0, 0, 0].item()
        expected = 1.0 + ppo.gamma * stored_values[0, 0].item()
        assert abs(stored_reward_env0 - expected) < 1e-5

        # Env 1 should have raw reward only
        stored_reward_env1 = ppo.storage.rewards[0, 1, 0].item()
        assert abs(stored_reward_env1 - 1.0) < 1e-5


class TestNormalizationUpdates:
    """Tests for observation-normalization update timing."""

    def test_normalization_uses_rollout_after_policy_update(self) -> None:
        """Normalization statistics should stay fixed during a rollout and update from stored observations."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS, obs_normalization=True)
        critic = _make_critic(obs, obs_groups, obs_normalization=True)
        storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
        ppo = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1, schedule="fixed")

        for step in range(NUM_STEPS):
            rollout_obs = TensorDict(
                {"policy": torch.full((NUM_ENVS, OBS_DIM), float(step + 1))}, batch_size=[NUM_ENVS]
            )
            next_obs = TensorDict({"policy": torch.full((NUM_ENVS, OBS_DIM), float(step + 2))}, batch_size=[NUM_ENVS])
            ppo.act(rollout_obs)
            ppo.process_env_step(next_obs, torch.ones(NUM_ENVS), torch.zeros(NUM_ENVS), {})

        assert actor.obs_normalizer.count == 0
        assert critic.obs_normalizer.count == 0

        actor(storage.observations[0], stochastic_output=True)
        current_log_prob = actor.get_output_log_prob(storage.actions[0])
        ratio = torch.exp(current_log_prob - storage.actions_log_prob[0].squeeze(-1))
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)

        expected_mean = storage.observations["policy"].flatten(0, 1).mean(dim=0)
        ppo.compute_returns(next_obs)
        ppo.update()

        expected_count = NUM_ENVS * NUM_STEPS
        assert actor.obs_normalizer.count == expected_count
        assert critic.obs_normalizer.count == expected_count
        assert torch.allclose(actor.obs_normalizer.mean, expected_mean)
        assert torch.allclose(critic.obs_normalizer.mean, expected_mean)


class TestPPOLosses:
    """Tests for PPO loss computation correctness."""

    def test_recurrent_surrogate_matches_per_transition_objective(self) -> None:
        """A recurrent minibatch must pair each action with its own old probability and advantage."""
        torch.manual_seed(7)
        num_envs = 1
        obs = make_obs(num_envs, 3)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        rnn_cfg = {"hidden_dims": [8], "rnn_hidden_dim": 8, "rnn_type": "gru"}
        actor = RNNModel(
            obs, obs_groups, "actor", 2, distribution_cfg={"class_name": "GaussianDistribution"}, **rnn_cfg
        )
        critic = RNNModel(obs, obs_groups, "critic", 1, **rnn_cfg)
        storage = RolloutStorage("rl", num_envs, 6, obs, [2])
        ppo = PPO(
            actor,
            critic,
            storage,
            num_learning_epochs=1,
            num_mini_batches=1,
            schedule="fixed",
            learning_rate=0.0,
            value_loss_coef=0.0,
            entropy_coef=0.0,
            max_grad_norm=1e6,
        )
        with torch.inference_mode():
            for step in range(storage.num_transitions_per_env):
                ppo.act(obs)
                obs = make_obs(num_envs, 3)
                dones = torch.full((num_envs,), step == 2)
                ppo.process_env_step(obs, torch.randn(num_envs), dones, {})
            ppo.compute_returns(obs)

        reference = copy.deepcopy(actor)
        batch = next(storage.recurrent_mini_batch_generator(1, 1))
        reference(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
        log_prob = (
            torch.distributions.Normal(reference.output_mean, reference.output_std)
            .log_prob(batch.actions)
            .sum(-1)
            .flatten()
        )
        ratios = (log_prob - batch.old_actions_log_prob.flatten()).exp()
        advantages = batch.advantages.flatten().clone()
        expected_loss = -torch.minimum(
            ratios * advantages, ratios.clamp(1.0 - ppo.clip_param, 1.0 + ppo.clip_param) * advantages
        ).mean()
        expected_loss.backward()

        losses = ppo.update()
        torch.testing.assert_close(losses["surrogate"], expected_loss.item(), atol=1e-6, rtol=1e-5)
        for actual, expected in zip(actor.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)

    @pytest.mark.parametrize("advantage, ratio, expected", [(1.0, 1.5, -1.2), (-1.0, 0.5, 0.8), (1.0, 0.5, -0.5)])
    def test_surrogate_loss_clipping(self, advantage: float, ratio: float, expected: float) -> None:
        """The update clips probability ratios for positive and negative advantages."""
        ppo, obs = _build_ppo(num_learning_epochs=1, num_mini_batches=1, learning_rate=0.0)
        _fill_rollout(ppo, obs)
        ppo.storage.actions_log_prob.sub_(torch.log(torch.tensor(ratio)))
        ppo.storage.advantages.fill_(advantage)

        losses = ppo.update()

        assert losses["surrogate"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "clipped, return_offset, expected", [(True, 0.0, 0.64), (False, 0.0, 0.0), (True, -1.0, 1.0)]
    )
    def test_value_loss_clipping(self, clipped: bool, return_offset: float, expected: float) -> None:
        """The update penalizes changes beyond the value clip only when enabled."""
        ppo, obs = _build_ppo(
            num_learning_epochs=1, num_mini_batches=1, learning_rate=0.0, use_clipped_value_loss=clipped
        )
        _fill_rollout(ppo, obs)
        ppo.storage.returns.copy_(ppo.storage.values + return_offset)
        ppo.storage.values.sub_(1.0)

        losses = ppo.update()

        assert losses["value"] == pytest.approx(expected, abs=1e-6)


class TestAdaptiveLearningRate:
    """Tests for adaptive KL-based learning rate scheduling."""

    @pytest.mark.parametrize(
        "kl, initial_lr, expected_lr",
        [
            (0.03, 1e-3, 1e-3 / 1.5),
            (0.002, 1e-3, 1.5e-3),
            (0.01, 1e-3, 1e-3),
            (0.0, 1e-3, 1e-3),
            (0.03, 1.2e-5, 1e-5),
            (0.002, 8e-3, 1e-2),
        ],
    )
    def test_update_adapts_optimizer_lr(
        self, monkeypatch: pytest.MonkeyPatch, kl: float, initial_lr: float, expected_lr: float
    ) -> None:
        """The update applies the KL-based learning rate to every optimizer group."""
        ppo, obs = _build_ppo(num_learning_epochs=1, num_mini_batches=1, schedule="adaptive", learning_rate=initial_lr)
        _fill_rollout(ppo, obs)
        monkeypatch.setattr(ppo.actor, "get_kl_divergence", lambda *_: torch.tensor(kl))

        ppo.update()

        assert ppo.learning_rate == pytest.approx(expected_lr)
        assert all(group["lr"] == pytest.approx(expected_lr) for group in ppo.optimizer.param_groups)


class TestMixedPrecision:
    """Tests for the use_mixed_precision flag."""

    def test_flag_defaults_to_false(self) -> None:
        """Mixed precision must be opt-in."""
        ppo, _obs = _build_ppo()
        assert ppo.use_mixed_precision is False

    def test_update_runs_with_mixed_precision(self) -> None:
        """update() with the flag on returns finite losses and changes parameters."""
        ppo, obs = _build_ppo(use_mixed_precision=True, schedule="fixed")
        ppo.train_mode()
        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        ppo.compute_returns(obs)

        before = [p.clone() for p in ppo.actor.parameters()]
        loss_dict = ppo.update()

        assert all(torch.isfinite(torch.tensor(v)) for v in loss_dict.values())
        after = list(ppo.actor.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after)), "actor params should change"


class TestCheckpoint:
    """Tests for saving and loading PPO checkpoints."""

    def test_rnd_weight_schedule_resumes(self) -> None:
        """Loading a checkpoint should continue the RND weight schedule where it stopped."""
        source, obs = _build_ppo(rnd_cfg=RND_CFG)
        for _ in range(3):
            source.rnd.get_intrinsic_reward(obs)
        restored, _ = _build_ppo(rnd_cfg=RND_CFG)
        restored.load(source.save(), load_cfg=None, strict=True)
        assert restored.rnd.update_counter == 3
        restored.rnd.get_intrinsic_reward(obs)
        assert restored.rnd.weight == 0.25

    def test_legacy_checkpoint_without_rnd_counter_loads(self) -> None:
        """Checkpoints saved before the RND counter was stored should still load."""
        source, _ = _build_ppo(rnd_cfg=RND_CFG)
        checkpoint = source.save()
        del checkpoint["rnd_update_counter"]
        restored, _ = _build_ppo(rnd_cfg=RND_CFG)
        restored.load(checkpoint, load_cfg=None, strict=True)
        assert restored.rnd.update_counter == 0


class TestSharedCNNEncoders:
    """Tests for actor and critic sharing CNN encoders."""

    def test_shared_parameters_are_stepped_once(self) -> None:
        """Each shared parameter should appear once in the optimizer and receive a single Adam step."""
        ppo = _build_ppo_with_shared_cnn()
        params = ppo.optimizer.param_groups[0]["params"]
        assert len(params) == len({id(p) for p in params})

        shared = next(ppo.actor.cnns.parameters())
        before = shared.detach().clone()
        for p in params:
            p.grad = torch.ones_like(p)
        ppo.optimizer.step()
        assert ppo.optimizer.state[shared]["step"] == 1
        # The first Adam step moves each parameter by the learning rate
        torch.testing.assert_close((before - shared.detach()).abs().max().item(), 1e-3, atol=1e-6, rtol=0)
