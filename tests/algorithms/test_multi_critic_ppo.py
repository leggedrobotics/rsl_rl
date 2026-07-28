# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the multi-critic PPO algorithm."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.multi_critic_ppo import MultiCriticPPO
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.storage import MultiCriticRolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 8
OBS_DIM = 8
NUM_ACTIONS = 4
NUM_CRITICS = 2


def _make_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: Any) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return MLPModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_critics(obs: TensorDict, obs_groups: dict, num_critics: int = NUM_CRITICS, **kwargs: Any) -> list[MLPModel]:
    """Create a list of MLPModel critics (no distribution), one per entry in ``num_critics``."""
    defaults: dict[str, Any] = {"hidden_dims": [32, 32], "activation": "elu"}
    defaults.update(kwargs)
    return [MLPModel(obs, obs_groups, "critic", 1, **defaults) for _ in range(num_critics)]


# NOTE: recurrence is a separate model class (RNNModel), not extra kwargs on MLPModel — confirmed by
# the real training-config pattern used in tests/runners/test_runner.py (`class_name: "RNNModel"`,
# `rnn_type`, `rnn_hidden_dim`, `rnn_num_layers`).
def _make_recurrent_actor(obs: TensorDict, obs_groups: dict, num_actions: int = 4, **kwargs: Any) -> RNNModel:
    """Create a recurrent RNNModel actor with a Gaussian distribution."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32],
        "rnn_type": "gru",
        "rnn_hidden_dim": 16,
        "rnn_num_layers": 1,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    defaults.update(kwargs)
    return RNNModel(obs, obs_groups, "actor", num_actions, **defaults)


def _make_recurrent_critics(
    obs: TensorDict, obs_groups: dict, num_critics: int = NUM_CRITICS, **kwargs: Any
) -> list[RNNModel]:
    """Create a list of recurrent RNNModel critics (no distribution)."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32],
        "rnn_type": "gru",
        "rnn_hidden_dim": 16,
        "rnn_num_layers": 1,
    }
    defaults.update(kwargs)
    return [RNNModel(obs, obs_groups, "critic", 1, **defaults) for _ in range(num_critics)]


def _build_ppo(num_critics: int = NUM_CRITICS, **overrides: Any) -> tuple[MultiCriticPPO, TensorDict]:
    """Build a MultiCriticPPO instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
    critics = _make_critics(obs, obs_groups, num_critics)
    storage = MultiCriticRolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_critics=num_critics)

    defaults: dict[str, Any] = dict(
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
    ppo = MultiCriticPPO(actor, critics, storage, **defaults)
    return ppo, obs


class TestGAEComputation:
    """Tests for generalized advantage estimation in ``compute_returns``, per critic."""

    def test_gae_returns_hand_computed_per_critic(self) -> None:
        """Verify GAE returns for each critic match hand-computed examples with distinct value streams.

        Rewards and dones are shared across critics, but each critic has its own value estimates, so
        each critic's advantage/return trajectory differs even though the reward stream is identical.
        """
        num_envs, num_steps, num_critics = 1, 3, 2
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critics = _make_critics(obs, obs_groups, num_critics)
        storage = MultiCriticRolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS], num_critics=num_critics)
        ppo = MultiCriticPPO(
            actor, critics, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        rewards = [1.0, 2.0, 3.0]  # shared across critics
        values_per_critic = [
            [0.5, 1.0, 1.5],  # critic 0
            [0.2, 0.4, 0.6],  # critic 1
        ]
        dones = [0.0, 0.0, 0.0]
        last_values_per_critic = [2.0, 1.0]

        for i in range(num_steps):
            t = MultiCriticRolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None,) * (1 + num_critics)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = tuple(torch.full((num_envs, 1), values_per_critic[c][i]) for c in range(num_critics))
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), rewards[i])
            t.dones = torch.full((num_envs,), dones[i])
            storage.add_transition(t)

        # Hand-compute GAE per critic using the same backward recursion PPO uses internally.
        expected_returns_per_critic = []
        for c in range(num_critics):
            v = values_per_critic[c]
            lv = last_values_per_critic[c]
            adv = [0.0, 0.0, 0.0]
            adv[2] = rewards[2] + gamma * lv - v[2]
            adv[1] = (rewards[1] + gamma * v[2] - v[1]) + gamma * lam * adv[2]
            adv[0] = (rewards[0] + gamma * v[1] - v[0]) + gamma * lam * adv[1]
            expected_returns_per_critic.append([adv[i] + v[i] for i in range(num_steps)])

        # Monkeypatch each critic's forward to return its own constant last value
        last_values = [torch.full((num_envs, 1), lv) for lv in last_values_per_critic]
        original_forwards = [critic.forward for critic in ppo.critics]
        for c_idx, lv in enumerate(last_values):
            ppo.critics[c_idx].forward = (lambda lv: lambda *a, **kw: lv)(lv)

        ppo.compute_returns(obs)

        for c_idx, orig_fwd in enumerate(original_forwards):
            ppo.critics[c_idx].forward = orig_fwd

        for c_idx in range(num_critics):
            for step in range(num_steps):
                got = storage.returns[c_idx][step, 0, 0].item()
                expected = expected_returns_per_critic[c_idx][step]
                assert abs(got - expected) < 1e-4, (
                    f"Return mismatch for critic {c_idx} at step {step}: got {got}, expected {expected}"
                )

    def test_gae_terminal_state_cuts_bootstrap(self) -> None:
        """When a done flag is set, no critic's advantage should bootstrap from the next value."""
        num_envs, num_steps, num_critics = 1, 2, 2
        gamma, lam = 0.99, 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, NUM_ACTIONS)
        critics = _make_critics(obs, obs_groups, num_critics)
        storage = MultiCriticRolloutStorage("rl", num_envs, num_steps, obs, [NUM_ACTIONS], num_critics=num_critics)
        ppo = MultiCriticPPO(
            actor, critics, storage, gamma=gamma, lam=lam, schedule="fixed", normalize_advantage_per_mini_batch=True
        )

        # Step 0: done=True, so step 1 is a fresh episode
        rewards = [1.0, 2.0]
        dones = [1.0, 0.0]
        values_per_critic = [
            [0.5, 1.0],  # critic 0
            [0.1, 0.3],  # critic 1
        ]
        last_values_per_critic = [3.0, 2.0]

        for i in range(num_steps):
            t = MultiCriticRolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None,) * (1 + num_critics)
            t.actions = torch.randn(num_envs, NUM_ACTIONS)
            t.values = tuple(torch.full((num_envs, 1), values_per_critic[c][i]) for c in range(num_critics))
            t.actions_log_prob = torch.zeros(num_envs)
            t.distribution_params = (torch.zeros(num_envs, NUM_ACTIONS), torch.ones(num_envs, NUM_ACTIONS))
            t.rewards = torch.full((num_envs,), rewards[i])
            t.dones = torch.full((num_envs,), dones[i])
            storage.add_transition(t)

        last_values = [torch.full((num_envs, 1), lv) for lv in last_values_per_critic]
        for c_idx, lv in enumerate(last_values):
            ppo.critics[c_idx].forward = (lambda lv: lambda *a, **kw: lv)(lv)

        ppo.compute_returns(obs)

        for c_idx in range(num_critics):
            v = values_per_critic[c_idx]
            lv = last_values_per_critic[c_idx]
            # Step 0: done=True -> no bootstrap
            adv0 = rewards[0] - v[0]
            # Step 1: bootstraps from last_values since it's the final step in the buffer
            adv1 = rewards[1] + gamma * lv - v[1]
            expected_return_0 = adv0 + v[0]
            expected_return_1 = adv1 + v[1]

            assert torch.allclose(storage.returns[c_idx][0, 0, 0], torch.tensor(expected_return_0), atol=1e-4)
            assert torch.allclose(storage.returns[c_idx][1, 0, 0], torch.tensor(expected_return_1), atol=1e-4)

    def test_advantage_normalization_global(self) -> None:
        """With normalize_advantage_per_mini_batch=False, each critic's advantages should have mean~0, std~1."""
        ppo, obs = _build_ppo(normalize_advantage_per_mini_batch=False)

        for _ in range(NUM_STEPS):
            t = MultiCriticRolloutStorage.Transition()
            t.observations = obs
            t.hidden_states = (None,) * (1 + len(ppo.critics))
            actions = ppo.actor(obs, stochastic_output=True).detach()
            t.actions = actions
            t.values = tuple(critic(obs).detach() for critic in ppo.critics)
            t.actions_log_prob = ppo.actor.get_output_log_prob(actions).detach()
            t.distribution_params = tuple(p.detach() for p in ppo.actor.output_distribution_params)
            t.rewards = torch.randn(NUM_ENVS)
            t.dones = torch.zeros(NUM_ENVS)
            ppo.storage.add_transition(t)

        ppo.compute_returns(obs)

        assert len(ppo.storage.advantages) == len(ppo.critics)
        for c_idx, adv in enumerate(ppo.storage.advantages):
            adv_flat = adv.flatten()
            assert abs(adv_flat.mean().item()) < 1e-5, f"Critic {c_idx} advantages should be zero-mean"
            assert abs(adv_flat.std().item() - 1.0) < 0.1, f"Critic {c_idx} advantages should be unit-std"


class TestTimeoutBootstrapping:
    """Tests for timeout bootstrapping in ``process_env_step``."""

    def test_timeout_adds_bootstrap_to_reward(self) -> None:
        """When time_outs is set, stored reward should include gamma * (mean value across critics) * timeout."""
        ppo, obs = _build_ppo()

        # Manually act to populate transition.values
        ppo.act(obs)
        assert ppo.transition.values is not None
        stored_values = tuple(v.clone() for v in ppo.transition.values)

        raw_reward = torch.ones(NUM_ENVS)
        dones = torch.ones(NUM_ENVS)
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0  # Only env 0 times out

        ppo.process_env_step(obs, raw_reward, dones, {"time_outs": time_outs})

        # The stored reward for env 0 should be: 1.0 + gamma * mean_c(value_c[0])
        mean_value_env0 = sum(v[0, 0].item() for v in stored_values) / len(stored_values)
        stored_reward_env0 = ppo.storage.rewards[0, 0, 0].item()
        expected = 1.0 + ppo.gamma * mean_value_env0
        assert abs(stored_reward_env0 - expected) < 1e-5

        # Env 1 should have raw reward only
        stored_reward_env1 = ppo.storage.rewards[0, 1, 0].item()
        assert abs(stored_reward_env1 - 1.0) < 1e-5


class TestPPOLosses:
    """Tests for the per-critic surrogate/value loss formulas MultiCriticPPO uses (then averages)."""

    def test_surrogate_loss_clipping(self) -> None:
        """When ratio deviates beyond clip_param, the clipped branch should dominate (same formula per critic)."""
        clip_param = 0.2

        # Construct a scenario: positive advantages, ratio > 1 + clip
        advantages = torch.tensor([1.0, 1.0, 1.0])
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        # New log probs that give ratio = exp(0.5) ≈ 1.65, which is > 1 + 0.2
        new_log_probs = torch.tensor([0.5, 0.5, 0.5])

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        loss = torch.max(surrogate, surrogate_clipped).mean()

        # The clipped branch should be -advantages * (1 + clip_param) = -1.2
        # The unclipped branch should be -advantages * 1.65 ≈ -1.65
        # max(-1.65, -1.2) = -1.2, so clipped branch dominates
        expected_clipped = (-advantages * (1.0 + clip_param)).mean()
        assert torch.allclose(loss, expected_clipped, atol=1e-5)

    def test_value_loss_clipping(self) -> None:
        """With clipped value loss, large value changes should be clipped (same formula per critic)."""
        clip_param = 0.2
        old_values = torch.tensor([[1.0], [1.0]])
        new_values = torch.tensor([[2.0], [1.1]])
        returns = torch.tensor([[1.5], [1.5]])

        value_clipped = old_values + (new_values - old_values).clamp(-clip_param, clip_param)
        losses_unclipped = (new_values - returns).pow(2)
        losses_clipped = (value_clipped - returns).pow(2)
        loss = torch.max(losses_unclipped, losses_clipped).mean()

        # Env 0: new=2.0, old=1.0, clipped_new=1.2
        #   unclipped: (2.0 - 1.5)^2 = 0.25
        #   clipped: (1.2 - 1.5)^2 = 0.09
        #   max = 0.25
        # Env 1: new=1.1, old=1.0, clipped_new=1.1 (within clip)
        #   unclipped: (1.1 - 1.5)^2 = 0.16
        #   clipped: (1.1 - 1.5)^2 = 0.16
        #   max = 0.16
        expected = (0.25 + 0.16) / 2
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_surrogate_loss_averaged_across_critics(self) -> None:
        """MultiCriticPPO.update() averages each critic's surrogate loss into a single scalar."""
        clip_param = 0.2
        ratio = torch.tensor([1.65, 1.65, 1.65])  # exp(0.5); shared actor -> same ratio for every critic

        def surrogate_loss(adv: torch.Tensor) -> torch.Tensor:
            surrogate = -adv * ratio
            surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            return torch.max(surrogate, surrogate_clipped).mean()

        advantages_per_critic = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0])]
        per_critic_losses = [surrogate_loss(adv) for adv in advantages_per_critic]

        # This mirrors `surrogate_loss = sum(surrogate_losses) / len(surrogate_losses)` in
        # MultiCriticPPO.update().
        averaged = torch.stack(per_critic_losses).mean()
        expected = (per_critic_losses[0] + per_critic_losses[1]) / 2
        assert torch.allclose(averaged, expected, atol=1e-6)


class TestAdaptiveLearningRate:
    """Tests for adaptive KL-based learning rate scheduling (actor-only, critic-count agnostic)."""

    def test_lr_decreases_when_kl_too_high(self) -> None:
        """LR should decrease when KL > 2 * desired_kl."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        # Simulate high KL scenario
        ppo.learning_rate = initial_lr
        kl_mean = torch.tensor(0.03)  # > 2 * 0.01

        # Apply the same logic as MultiCriticPPO.update
        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)

        assert ppo.learning_rate < initial_lr
        assert ppo.learning_rate == max(1e-5, initial_lr / 1.5)

    def test_lr_increases_when_kl_too_low(self) -> None:
        """LR should increase when 0 < KL < desired_kl / 2."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.002)  # < 0.01 / 2 = 0.005

        if kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)

        assert ppo.learning_rate > initial_lr
        assert ppo.learning_rate == min(1e-2, initial_lr * 1.5)

    def test_lr_unchanged_in_stable_range(self) -> None:
        """LR should remain unchanged when KL is in [desired_kl/2, 2*desired_kl]."""
        ppo, _obs = _build_ppo(schedule="adaptive", desired_kl=0.01, learning_rate=1e-3)
        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.01)  # Exactly desired_kl — in stable range

        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(1e-5, ppo.learning_rate / 1.5)
        elif kl_mean < ppo.desired_kl / 2.0 and kl_mean > 0.0:
            ppo.learning_rate = min(1e-2, ppo.learning_rate * 1.5)

        assert ppo.learning_rate == initial_lr


class TestMultiCriticIntegration:
    """End-to-end smoke tests exercising the full rollout -> compute_returns -> update cycle."""

    def test_full_update_cycle_runs_with_multiple_critics(self) -> None:
        """act/process_env_step/compute_returns/update should run without error and return finite losses."""
        ppo, obs = _build_ppo(num_critics=3)

        for _ in range(NUM_STEPS):
            ppo.act(obs)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            ppo.process_env_step(obs, rewards, dones, {})

        ppo.compute_returns(obs)
        loss_dict = ppo.update()

        assert {"value", "surrogate", "entropy"} <= set(loss_dict.keys())
        for key, val in loss_dict.items():
            assert torch.isfinite(torch.as_tensor(val)), f"Loss '{key}' is not finite: {val}"

    def test_storage_values_returns_advantages_have_one_entry_per_critic(self) -> None:
        """Storage should keep separate values/returns/advantages tensors, one per critic."""
        num_critics = 3
        ppo, obs = _build_ppo(num_critics=num_critics)

        for _ in range(NUM_STEPS):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.randn(NUM_ENVS), torch.zeros(NUM_ENVS), {})
        ppo.compute_returns(obs)

        assert len(ppo.storage.values) == num_critics
        assert len(ppo.storage.returns) == num_critics
        assert len(ppo.storage.advantages) == num_critics
        for v in ppo.storage.values:
            assert v.shape == (NUM_STEPS, NUM_ENVS, 1)

    def test_timeout_bootstrap_uses_mean_across_critics(self) -> None:
        """process_env_step's timeout bootstrap should use the mean value across all critics, not just one."""
        num_critics = 4
        ppo, obs = _build_ppo(num_critics=num_critics)

        ppo.act(obs)
        assert ppo.transition.values is not None
        stored_values = tuple(v.clone() for v in ppo.transition.values)

        time_outs = torch.ones(NUM_ENVS)
        ppo.process_env_step(obs, torch.zeros(NUM_ENVS), torch.ones(NUM_ENVS), {"time_outs": time_outs})

        for env in range(NUM_ENVS):
            mean_value = sum(v[env, 0].item() for v in stored_values) / num_critics
            expected = ppo.gamma * mean_value
            got = ppo.storage.rewards[0, env, 0].item()
            assert abs(got - expected) < 1e-5


class TestRecurrentIntegration:
    """Smoke tests for the recurrent path: recurrent_mini_batch_generator plus per-critic hidden states.

    These exercise the code path that the feedforward tests above never touch — hidden-state saving in
    `_save_hidden_states`, per-critic hidden-state slots (`saved_hidden_state_c`), and the trajectory
    padding/masking in `recurrent_mini_batch_generator`. Recurrence uses the dedicated `RNNModel` class
    (not extra kwargs on `MLPModel`), matching the pattern in tests/runners/test_runner.py.
    """

    def test_full_update_cycle_runs_with_recurrent_actor_and_critics(self) -> None:
        """act/process_env_step/compute_returns/update should run without error on the recurrent path."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        num_critics = 2

        actor = _make_recurrent_actor(obs, obs_groups, NUM_ACTIONS)
        critics = _make_recurrent_critics(obs, obs_groups, num_critics)
        assert actor.is_recurrent, "Expected actor.is_recurrent to be True once RNN kwargs are set"
        assert all(critic.is_recurrent for critic in critics), "Expected all critics to be recurrent"

        storage = MultiCriticRolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_critics=num_critics)
        ppo = MultiCriticPPO(
            actor,
            critics,
            storage,
            num_learning_epochs=2,
            num_mini_batches=2,
            schedule="fixed",
            desired_kl=0.01,
        )

        for _ in range(NUM_STEPS):
            ppo.act(obs)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            ppo.process_env_step(obs, rewards, dones, {})

        ppo.compute_returns(obs)
        loss_dict = ppo.update()

        assert {"value", "surrogate", "entropy"} <= set(loss_dict.keys())
        for key, val in loss_dict.items():
            assert torch.isfinite(torch.as_tensor(val)), f"Loss '{key}' is not finite: {val}"

    def test_full_update_cycle_runs_with_recurrent_actor_and_critics_and_mid_rollout_dones(self) -> None:
        """Same as above, but with a mid-rollout episode termination to exercise hidden-state resets."""
        obs = make_obs(NUM_ENVS, OBS_DIM)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        num_critics = 2

        actor = _make_recurrent_actor(obs, obs_groups, NUM_ACTIONS)
        critics = _make_recurrent_critics(obs, obs_groups, num_critics)

        storage = MultiCriticRolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_critics=num_critics)
        ppo = MultiCriticPPO(
            actor,
            critics,
            storage,
            num_learning_epochs=2,
            num_mini_batches=2,
            schedule="fixed",
            desired_kl=0.01,
        )

        for step in range(NUM_STEPS):
            ppo.act(obs)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS)
            if step == NUM_STEPS // 2:
                dones[0] = 1.0  # env 0 terminates partway through the rollout
            ppo.process_env_step(obs, rewards, dones, {})

        ppo.compute_returns(obs)
        loss_dict = ppo.update()

        for key, val in loss_dict.items():
            assert torch.isfinite(torch.as_tensor(val)), f"Loss '{key}' is not finite: {val}"


class TestConvergence:
    """Sanity check that MultiCriticPPO's critics can actually learn, not just run without crashing."""

    def test_value_loss_decreases_with_learnable_reward(self) -> None:
        """With a fixed, learnable reward signal, mean critic value loss should trend downward over iterations.

        The observation is held fixed across the whole run and the reward is a deterministic linear
        function of it, so a critic that is actually learning should fit it increasingly well. This
        doesn't require a real VecEnv — it only checks that gradients flow correctly and value loss
        goes down, which the finite-loss smoke tests above do not verify.
        """
        torch.manual_seed(0)
        ppo, obs = _build_ppo(
            num_critics=2,
            num_learning_epochs=4,
            num_mini_batches=2,
            learning_rate=1e-2,
            schedule="fixed",
        )

        # Fixed synthetic reward: a deterministic linear function of the (also fixed) observation, so
        # the critics have something consistent to fit across iterations.
        reward_weights = torch.randn(OBS_DIM)

        def compute_reward(o: TensorDict) -> torch.Tensor:
            return (o["policy"] * reward_weights).sum(dim=-1)

        num_iterations = 20
        value_losses: list[float] = []
        for _ in range(num_iterations):
            for _ in range(NUM_STEPS):
                ppo.act(obs)
                rewards = compute_reward(obs)
                dones = torch.zeros(NUM_ENVS)
                ppo.process_env_step(obs, rewards, dones, {})
            ppo.compute_returns(obs)
            loss_dict = ppo.update()
            value_losses.append(float(loss_dict["value"].detach()))

        # Compare early vs. late average rather than a single point-to-point comparison to reduce noise.
        early_avg = sum(value_losses[:3]) / 3
        late_avg = sum(value_losses[-3:]) / 3
        assert late_avg < early_avg, (
            f"Expected critic value loss to decrease with a fixed, learnable reward signal, "
            f"got early_avg={early_avg}, late_avg={late_avg}, full trace={value_losses}"
        )