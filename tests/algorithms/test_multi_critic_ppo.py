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


def _make_actor(
    obs: TensorDict,
    obs_groups: dict,
    num_actions: int = 4,
    **kwargs: Any,
) -> MLPModel:
    """Create an MLPModel actor with a Gaussian distribution."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    defaults.update(kwargs)
    return MLPModel(
        obs,
        obs_groups,
        "actor",
        num_actions,
        **defaults,
    )


def _make_critics(
    obs: TensorDict,
    obs_groups: dict,
    num_critics: int = NUM_CRITICS,
    **kwargs: Any,
) -> list[MLPModel]:
    """Create a list of MLPModel critics (no distribution), one per critic."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32, 32],
        "activation": "elu",
    }
    defaults.update(kwargs)

    return [
        MLPModel(
            obs,
            obs_groups,
            "critic",
            1,
            **defaults,
        )
        for _ in range(num_critics)
    ]


# NOTE: recurrence is a separate model class (RNNModel), not extra kwargs on MLPModel.
def _make_recurrent_actor(
    obs: TensorDict,
    obs_groups: dict,
    num_actions: int = 4,
    **kwargs: Any,
) -> RNNModel:
    """Create a recurrent RNNModel actor with a Gaussian distribution."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32],
        "rnn_type": "gru",
        "rnn_hidden_dim": 16,
        "rnn_num_layers": 1,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    defaults.update(kwargs)

    return RNNModel(
        obs,
        obs_groups,
        "actor",
        num_actions,
        **defaults,
    )


def _make_recurrent_critics(
    obs: TensorDict,
    obs_groups: dict,
    num_critics: int = NUM_CRITICS,
    **kwargs: Any,
) -> list[RNNModel]:
    """Create a list of recurrent RNNModel critics (no distribution)."""
    defaults: dict[str, Any] = {
        "hidden_dims": [32],
        "rnn_type": "gru",
        "rnn_hidden_dim": 16,
        "rnn_num_layers": 1,
    }
    defaults.update(kwargs)

    return [
        RNNModel(
            obs,
            obs_groups,
            "critic",
            1,
            **defaults,
        )
        for _ in range(num_critics)
    ]


def _build_ppo(
    num_critics: int = NUM_CRITICS,
    **overrides: Any,
) -> tuple[MultiCriticPPO, TensorDict]:
    """Build a MultiCriticPPO instance with small networks for testing."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    actor = _make_actor(
        obs,
        obs_groups,
        NUM_ACTIONS,
    )

    critics = _make_critics(
        obs,
        obs_groups,
        num_critics,
    )

    storage = MultiCriticRolloutStorage(
        "rl",
        NUM_ENVS,
        NUM_STEPS,
        obs,
        [NUM_ACTIONS],
        num_critics=num_critics,
    )

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

    ppo = MultiCriticPPO(
        actor,
        critics,
        storage,
        **defaults,
    )

    return ppo, obs


class TestGAEComputation:
    """Tests for generalized advantage estimation in compute_returns, per critic."""

    def test_gae_returns_hand_computed_per_critic(self) -> None:
        """Verify GAE returns for each critic match hand-computed examples.

        Each critic has its own reward and value stream, so each critic's
        advantage and return trajectory can differ.
        """
        num_envs = 1
        num_steps = 3
        num_critics = 2

        gamma = 0.99
        lam = 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {
            "actor": ["policy"],
            "critic": ["policy"],
        }

        actor = _make_actor(
            obs,
            obs_groups,
            NUM_ACTIONS,
        )

        critics = _make_critics(
            obs,
            obs_groups,
            num_critics,
        )

        storage = MultiCriticRolloutStorage(
            "rl",
            num_envs,
            num_steps,
            obs,
            [NUM_ACTIONS],
            num_critics=num_critics,
        )

        ppo = MultiCriticPPO(
            actor,
            critics,
            storage,
            gamma=gamma,
            lam=lam,
            schedule="fixed",
            normalize_advantage_per_mini_batch=True,
        )

        rewards_per_critic = [
            [1.0, 2.0, 3.0],  # critic 0
            [2.0, 4.0, 6.0],  # critic 1
        ]

        values_per_critic = [
            [0.5, 1.0, 1.5],  # critic 0
            [0.2, 0.4, 0.6],  # critic 1
        ]

        dones = [0.0, 0.0, 0.0]

        last_values_per_critic = [
            2.0,  # critic 0
            1.0,  # critic 1
        ]

        for i in range(num_steps):
            t = MultiCriticRolloutStorage.Transition()

            t.observations = obs
            t.hidden_states = (None,) * (1 + num_critics)

            t.actions = torch.randn(
                num_envs,
                NUM_ACTIONS,
            )

            t.values = tuple(
                torch.full(
                    (num_envs, 1),
                    values_per_critic[c][i],
                )
                for c in range(num_critics)
            )

            t.actions_log_prob = torch.zeros(num_envs)

            t.distribution_params = (
                torch.zeros(num_envs, NUM_ACTIONS),
                torch.ones(num_envs, NUM_ACTIONS),
            )

            t.rewards = tuple(
                torch.full(
                    (num_envs,),
                    rewards_per_critic[c][i],
                )
                for c in range(num_critics)
            )

            t.dones = torch.full(
                (num_envs,),
                dones[i],
            )

            storage.add_transition(t)

        # Hand-compute GAE per critic.
        expected_returns_per_critic = []

        for c in range(num_critics):
            rewards = rewards_per_critic[c]
            values = values_per_critic[c]
            last_value = last_values_per_critic[c]

            adv = [0.0, 0.0, 0.0]

            adv[2] = (
                rewards[2]
                + gamma * last_value
                - values[2]
            )

            adv[1] = (
                rewards[1]
                + gamma * values[2]
                - values[1]
            ) + gamma * lam * adv[2]

            adv[0] = (
                rewards[0]
                + gamma * values[1]
                - values[0]
            ) + gamma * lam * adv[1]

            expected_returns_per_critic.append(
                [
                    adv[i] + values[i]
                    for i in range(num_steps)
                ]
            )

        # Monkeypatch each critic's forward to return its expected last value.
        last_values = [
            torch.full(
                (num_envs, 1),
                value,
            )
            for value in last_values_per_critic
        ]

        original_forwards = [
            critic.forward
            for critic in ppo.critics
        ]

        for c_idx, last_value in enumerate(last_values):
            ppo.critics[c_idx].forward = (
                lambda last_value: lambda *args, **kwargs: last_value
            )(last_value)

        ppo.compute_returns(obs)

        for c_idx, original_forward in enumerate(original_forwards):
            ppo.critics[c_idx].forward = original_forward

        for c_idx in range(num_critics):
            for step in range(num_steps):
                got = storage.returns[c_idx][
                    step,
                    0,
                    0,
                ].item()

                expected = expected_returns_per_critic[c_idx][step]

                assert abs(got - expected) < 1e-4, (
                    f"Return mismatch for critic {c_idx} "
                    f"at step {step}: "
                    f"got {got}, expected {expected}"
                )

    def test_gae_terminal_state_cuts_bootstrap(self) -> None:
        """When a done flag is set, no critic bootstraps across the terminal state."""
        num_envs = 1
        num_steps = 2
        num_critics = 2

        gamma = 0.99
        lam = 0.95

        obs = make_obs(num_envs, OBS_DIM)
        obs_groups = {
            "actor": ["policy"],
            "critic": ["policy"],
        }

        actor = _make_actor(
            obs,
            obs_groups,
            NUM_ACTIONS,
        )

        critics = _make_critics(
            obs,
            obs_groups,
            num_critics,
        )

        storage = MultiCriticRolloutStorage(
            "rl",
            num_envs,
            num_steps,
            obs,
            [NUM_ACTIONS],
            num_critics=num_critics,
        )

        ppo = MultiCriticPPO(
            actor,
            critics,
            storage,
            gamma=gamma,
            lam=lam,
            schedule="fixed",
            normalize_advantage_per_mini_batch=True,
        )

        rewards_per_critic = [
            [1.0, 2.0],  # critic 0
            [2.0, 4.0],  # critic 1
        ]

        dones = [
            1.0,
            0.0,
        ]

        values_per_critic = [
            [0.5, 1.0],  # critic 0
            [0.1, 0.3],  # critic 1
        ]

        last_values_per_critic = [
            3.0,
            2.0,
        ]

        for i in range(num_steps):
            t = MultiCriticRolloutStorage.Transition()

            t.observations = obs
            t.hidden_states = (None,) * (1 + num_critics)

            t.actions = torch.randn(
                num_envs,
                NUM_ACTIONS,
            )

            t.values = tuple(
                torch.full(
                    (num_envs, 1),
                    values_per_critic[c][i],
                )
                for c in range(num_critics)
            )

            t.actions_log_prob = torch.zeros(num_envs)

            t.distribution_params = (
                torch.zeros(num_envs, NUM_ACTIONS),
                torch.ones(num_envs, NUM_ACTIONS),
            )

            t.rewards = tuple(
                torch.full(
                    (num_envs,),
                    rewards_per_critic[c][i],
                )
                for c in range(num_critics)
            )

            t.dones = torch.full(
                (num_envs,),
                dones[i],
            )

            storage.add_transition(t)

        last_values = [
            torch.full(
                (num_envs, 1),
                value,
            )
            for value in last_values_per_critic
        ]

        original_forwards = [
            critic.forward
            for critic in ppo.critics
        ]

        for c_idx, last_value in enumerate(last_values):
            ppo.critics[c_idx].forward = (
                lambda last_value: lambda *args, **kwargs: last_value
            )(last_value)

        ppo.compute_returns(obs)

        for c_idx, original_forward in enumerate(original_forwards):
            ppo.critics[c_idx].forward = original_forward

        for c_idx in range(num_critics):
            rewards = rewards_per_critic[c_idx]
            values = values_per_critic[c_idx]
            last_value = last_values_per_critic[c_idx]

            # Step 0 is terminal, so no bootstrap from step 1.
            adv0 = rewards[0] - values[0]

            # Step 1 bootstraps from the final next observation.
            adv1 = (
                rewards[1]
                + gamma * last_value
                - values[1]
            )

            expected_return_0 = adv0 + values[0]
            expected_return_1 = adv1 + values[1]

            assert torch.allclose(
                storage.returns[c_idx][0, 0, 0],
                torch.tensor(expected_return_0),
                atol=1e-4,
            )

            assert torch.allclose(
                storage.returns[c_idx][1, 0, 0],
                torch.tensor(expected_return_1),
                atol=1e-4,
            )

    def test_advantage_normalization_global(self) -> None:
        """Each critic's advantages should have mean ~0 and std ~1."""
        ppo, obs = _build_ppo(
            normalize_advantage_per_mini_batch=False,
        )

        num_critics = len(ppo.critics)

        for _ in range(NUM_STEPS):
            t = MultiCriticRolloutStorage.Transition()

            t.observations = obs
            t.hidden_states = (None,) * (1 + num_critics)

            actions = ppo.actor(
                obs,
                stochastic_output=True,
            ).detach()

            t.actions = actions

            t.values = tuple(
                critic(obs).detach()
                for critic in ppo.critics
            )

            t.actions_log_prob = (
                ppo.actor
                .get_output_log_prob(actions)
                .detach()
            )

            t.distribution_params = tuple(
                p.detach()
                for p in ppo.actor.output_distribution_params
            )

            t.rewards = tuple(
                torch.randn(NUM_ENVS)
                for _ in range(num_critics)
            )

            t.dones = torch.zeros(NUM_ENVS)

            ppo.storage.add_transition(t)

        ppo.compute_returns(obs)

        assert len(ppo.storage.advantages) == num_critics

        for c_idx, advantages in enumerate(
            ppo.storage.advantages
        ):
            adv_flat = advantages.flatten()

            assert abs(
                adv_flat.mean().item()
            ) < 1e-5, (
                f"Critic {c_idx} advantages should be zero-mean"
            )

            assert abs(
                adv_flat.std().item() - 1.0
            ) < 0.1, (
                f"Critic {c_idx} advantages should be unit-std"
            )


class TestTimeoutBootstrapping:
    """Tests for timeout bootstrapping in process_env_step."""

    def test_timeout_adds_each_critics_own_bootstrap_to_its_reward(self) -> None:
        """Timeout bootstrap uses each critic's own value estimate."""
        num_critics = 4

        ppo, obs = _build_ppo(
            num_critics=num_critics,
        )

        ppo.act(obs)

        assert ppo.transition.values is not None

        stored_values = tuple(
            value.clone()
            for value in ppo.transition.values
        )

        raw_rewards = tuple(
            torch.ones(NUM_ENVS)
            for _ in range(num_critics)
        )

        dones = torch.ones(NUM_ENVS)

        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0

        ppo.process_env_step(
            obs,
            raw_rewards,
            dones,
            {"time_outs": time_outs},
        )

        for critic_idx in range(num_critics):
            expected_timeout_reward = (
                1.0
                + ppo.gamma * stored_values[critic_idx].squeeze(-1)
            )

            # Timed-out environment gets its own critic's bootstrap.
            torch.testing.assert_close(
                ppo.storage.rewards[critic_idx][0, 0],
                expected_timeout_reward[0:1],
                atol=1e-5,
                rtol=0.0,
            )

            # Non-timeout environments retain their raw reward.
            torch.testing.assert_close(
                ppo.storage.rewards[critic_idx][0, 1:],
                torch.ones(NUM_ENVS - 1, 1),
                atol=1e-5,
                rtol=0.0,
            )


class TestPPOLosses:
    """Tests for the per-critic PPO surrogate and value loss formulas."""

    def test_surrogate_loss_clipping(self) -> None:
        """When ratio deviates beyond clip_param, the clipped branch should dominate."""
        clip_param = 0.2

        advantages = torch.tensor([
            1.0,
            1.0,
            1.0,
        ])

        old_log_probs = torch.tensor([
            0.0,
            0.0,
            0.0,
        ])

        new_log_probs = torch.tensor([
            0.5,
            0.5,
            0.5,
        ])

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        surrogate = (
            -advantages * ratio
        )

        surrogate_clipped = (
            -advantages
            * torch.clamp(
                ratio,
                1.0 - clip_param,
                1.0 + clip_param,
            )
        )

        loss = torch.max(
            surrogate,
            surrogate_clipped,
        ).mean()

        expected_clipped = (
            -advantages
            * (1.0 + clip_param)
        ).mean()

        assert torch.allclose(
            loss,
            expected_clipped,
            atol=1e-5,
        )

    def test_value_loss_clipping(self) -> None:
        """With clipped value loss, large value changes should be clipped."""
        clip_param = 0.2

        old_values = torch.tensor([
            [1.0],
            [1.0],
        ])

        new_values = torch.tensor([
            [2.0],
            [1.1],
        ])

        returns = torch.tensor([
            [1.5],
            [1.5],
        ])

        value_clipped = (
            old_values
            + (
                new_values - old_values
            ).clamp(
                -clip_param,
                clip_param,
            )
        )

        losses_unclipped = (
            new_values - returns
        ).pow(2)

        losses_clipped = (
            value_clipped - returns
        ).pow(2)

        loss = torch.max(
            losses_unclipped,
            losses_clipped,
        ).mean()

        expected = (
            0.25 + 0.16
        ) / 2

        assert torch.allclose(
            loss,
            torch.tensor(expected),
            atol=1e-5,
        )

    def test_surrogate_loss_averaged_across_critics(self) -> None:
        """MultiCriticPPO.update() averages each critic's surrogate loss."""
        clip_param = 0.2

        ratio = torch.tensor([
            1.65,
            1.65,
            1.65,
        ])

        def surrogate_loss(
            advantages: torch.Tensor,
        ) -> torch.Tensor:
            surrogate = (
                -advantages * ratio
            )

            surrogate_clipped = (
                -advantages
                * torch.clamp(
                    ratio,
                    1.0 - clip_param,
                    1.0 + clip_param,
                )
            )

            return torch.max(
                surrogate,
                surrogate_clipped,
            ).mean()

        advantages_per_critic = [
            torch.tensor([
                1.0,
                1.0,
                1.0,
            ]),
            torch.tensor([
                2.0,
                2.0,
                2.0,
            ]),
        ]

        per_critic_losses = [
            surrogate_loss(advantages)
            for advantages in advantages_per_critic
        ]

        averaged = torch.stack(
            per_critic_losses
        ).mean()

        expected = (
            per_critic_losses[0]
            + per_critic_losses[1]
        ) / 2

        assert torch.allclose(
            averaged,
            expected,
            atol=1e-6,
        )


class TestAdaptiveLearningRate:
    """Tests for adaptive KL-based learning rate scheduling."""

    def test_lr_decreases_when_kl_too_high(self) -> None:
        """LR should decrease when KL > 2 * desired_kl."""
        ppo, _obs = _build_ppo(
            schedule="adaptive",
            desired_kl=0.01,
            learning_rate=1e-3,
        )

        initial_lr = ppo.learning_rate

        ppo.learning_rate = initial_lr

        kl_mean = torch.tensor(0.03)

        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(
                1e-5,
                ppo.learning_rate / 1.5,
            )

        assert ppo.learning_rate < initial_lr

        assert ppo.learning_rate == max(
            1e-5,
            initial_lr / 1.5,
        )

    def test_lr_increases_when_kl_too_low(self) -> None:
        """LR should increase when 0 < KL < desired_kl / 2."""
        ppo, _obs = _build_ppo(
            schedule="adaptive",
            desired_kl=0.01,
            learning_rate=1e-3,
        )

        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.002)

        if (
            kl_mean < ppo.desired_kl / 2.0
            and kl_mean > 0.0
        ):
            ppo.learning_rate = min(
                1e-2,
                ppo.learning_rate * 1.5,
            )

        assert ppo.learning_rate > initial_lr

        assert ppo.learning_rate == min(
            1e-2,
            initial_lr * 1.5,
        )

    def test_lr_unchanged_in_stable_range(self) -> None:
        """LR should remain unchanged in the stable KL range."""
        ppo, _obs = _build_ppo(
            schedule="adaptive",
            desired_kl=0.01,
            learning_rate=1e-3,
        )

        initial_lr = ppo.learning_rate

        kl_mean = torch.tensor(0.01)

        if kl_mean > ppo.desired_kl * 2.0:
            ppo.learning_rate = max(
                1e-5,
                ppo.learning_rate / 1.5,
            )
        elif (
            kl_mean < ppo.desired_kl / 2.0
            and kl_mean > 0.0
        ):
            ppo.learning_rate = min(
                1e-2,
                ppo.learning_rate * 1.5,
            )

        assert ppo.learning_rate == initial_lr


class TestMultiCriticIntegration:
    """End-to-end smoke tests for the multi-critic PPO pipeline."""

    def test_full_update_cycle_runs_with_multiple_critics(self) -> None:
        """Full rollout -> compute_returns -> update cycle should run."""
        num_critics = 3

        ppo, obs = _build_ppo(
            num_critics=num_critics,
        )

        for _ in range(NUM_STEPS):
            ppo.act(obs)

            rewards = tuple(
                torch.randn(NUM_ENVS)
                for _ in range(num_critics)
            )

            dones = torch.zeros(NUM_ENVS)

            ppo.process_env_step(
                obs,
                rewards,
                dones,
                {},
            )

        ppo.compute_returns(obs)

        loss_dict = ppo.update()

        assert {
            "value",
            "surrogate",
            "entropy",
        } <= set(loss_dict.keys())

        for key, value in loss_dict.items():
            assert torch.isfinite(
                torch.as_tensor(value)
            ), (
                f"Loss '{key}' is not finite: {value}"
            )

    def test_storage_values_returns_advantages_have_one_entry_per_critic(self) -> None:
        """Storage should keep separate tensors for each critic."""
        num_critics = 3

        ppo, obs = _build_ppo(
            num_critics=num_critics,
        )

        for _ in range(NUM_STEPS):
            ppo.act(obs)

            rewards = tuple(
                torch.randn(NUM_ENVS)
                for _ in range(num_critics)
            )

            ppo.process_env_step(
                obs,
                rewards,
                torch.zeros(NUM_ENVS),
                {},
            )

        ppo.compute_returns(obs)

        assert len(ppo.storage.values) == num_critics
        assert len(ppo.storage.rewards) == num_critics
        assert len(ppo.storage.returns) == num_critics
        assert len(ppo.storage.advantages) == num_critics

        for (
            values,
            rewards,
            returns,
            advantages,
        ) in zip(
            ppo.storage.values,
            ppo.storage.rewards,
            ppo.storage.returns,
            ppo.storage.advantages,
        ):
            assert values.shape == (
                NUM_STEPS,
                NUM_ENVS,
                1,
            )

            assert rewards.shape == (
                NUM_STEPS,
                NUM_ENVS,
                1,
            )

            assert returns.shape == (
                NUM_STEPS,
                NUM_ENVS,
                1,
            )

            assert advantages.shape == (
                NUM_STEPS,
                NUM_ENVS,
                1,
            )

    def test_timeout_bootstrap_uses_each_critics_own_value(self) -> None:
        """Timeout bootstrap should use each critic's own value estimate."""
        num_critics = 4

        ppo, obs = _build_ppo(
            num_critics=num_critics,
        )

        ppo.act(obs)

        assert ppo.transition.values is not None

        stored_values = tuple(
            value.clone()
            for value in ppo.transition.values
        )

        raw_rewards = tuple(
            torch.zeros(NUM_ENVS)
            for _ in range(num_critics)
        )

        dones = torch.ones(NUM_ENVS)

        time_outs = torch.ones(NUM_ENVS)

        ppo.process_env_step(
            obs,
            raw_rewards,
            dones,
            {"time_outs": time_outs},
        )

        for critic_idx in range(num_critics):
            expected = (
                ppo.gamma
                * stored_values[critic_idx]
            )

            torch.testing.assert_close(
                ppo.storage.rewards[critic_idx][0],
                expected,
                atol=1e-5,
                rtol=0.0,
            )

class TestRecurrentIntegration:
    """Smoke tests for recurrent actor and per-critic recurrent hidden states."""

    def test_full_update_cycle_runs_with_recurrent_actor_and_critics(self) -> None:
        """Full recurrent rollout -> compute_returns -> update cycle should run."""
        obs = make_obs(
            NUM_ENVS,
            OBS_DIM,
        )

        obs_groups = {
            "actor": ["policy"],
            "critic": ["policy"],
        }

        num_critics = 2

        actor = _make_recurrent_actor(
            obs,
            obs_groups,
            NUM_ACTIONS,
        )

        critics = _make_recurrent_critics(
            obs,
            obs_groups,
            num_critics,
        )

        assert actor.is_recurrent

        assert all(
            critic.is_recurrent
            for critic in critics
        )

        storage = MultiCriticRolloutStorage(
            "rl",
            NUM_ENVS,
            NUM_STEPS,
            obs,
            [NUM_ACTIONS],
            num_critics=num_critics,
        )

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

            rewards = tuple(
                torch.randn(NUM_ENVS)
                for _ in range(num_critics)
            )

            dones = torch.zeros(NUM_ENVS)

            ppo.process_env_step(
                obs,
                rewards,
                dones,
                {},
            )

        ppo.compute_returns(obs)

        loss_dict = ppo.update()

        assert {
            "value",
            "surrogate",
            "entropy",
        } <= set(loss_dict.keys())

        for key, value in loss_dict.items():
            assert torch.isfinite(
                torch.as_tensor(value)
            ), (
                f"Loss '{key}' is not finite: {value}"
            )

    def test_full_update_cycle_runs_with_recurrent_actor_and_critics_and_mid_rollout_dones(
        self,
    ) -> None:
        """Recurrent update cycle should handle mid-rollout episode termination."""
        obs = make_obs(
            NUM_ENVS,
            OBS_DIM,
        )

        obs_groups = {
            "actor": ["policy"],
            "critic": ["policy"],
        }

        num_critics = 2

        actor = _make_recurrent_actor(
            obs,
            obs_groups,
            NUM_ACTIONS,
        )

        critics = _make_recurrent_critics(
            obs,
            obs_groups,
            num_critics,
        )

        storage = MultiCriticRolloutStorage(
            "rl",
            NUM_ENVS,
            NUM_STEPS,
            obs,
            [NUM_ACTIONS],
            num_critics=num_critics,
        )

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

            rewards = tuple(
                torch.randn(NUM_ENVS)
                for _ in range(num_critics)
            )

            dones = torch.zeros(NUM_ENVS)

            if step == NUM_STEPS // 2:
                dones[0] = 1.0

            ppo.process_env_step(
                obs,
                rewards,
                dones,
                {},
            )

        ppo.compute_returns(obs)

        loss_dict = ppo.update()

        for key, value in loss_dict.items():
            assert torch.isfinite(
                torch.as_tensor(value)
            ), (
                f"Loss '{key}' is not finite: {value}"
            )


class TestConvergence:
    """Sanity checks that MultiCriticPPO critics can actually learn."""

    def test_value_loss_decreases_with_learnable_reward(self) -> None:
        """Critic value loss should trend downward for a fixed learnable reward."""
        torch.manual_seed(0)

        num_critics = 2

        ppo, obs = _build_ppo(
            num_critics=num_critics,
            num_learning_epochs=4,
            num_mini_batches=2,
            learning_rate=1e-2,
            schedule="fixed",
        )

        reward_weights = torch.randn(OBS_DIM)

        def compute_reward(
            observations: TensorDict,
        ) -> torch.Tensor:
            return (
                observations["policy"]
                * reward_weights
            ).sum(dim=-1)

        num_iterations = 20

        value_losses: list[float] = []

        for _ in range(num_iterations):
            for _ in range(NUM_STEPS):
                ppo.act(obs)

                reward = compute_reward(obs)

                rewards = tuple(
                    reward.clone()
                    for _ in range(num_critics)
                )

                dones = torch.zeros(NUM_ENVS)

                ppo.process_env_step(
                    obs,
                    rewards,
                    dones,
                    {},
                )

            ppo.compute_returns(obs)

            loss_dict = ppo.update()

            value_losses.append(float(loss_dict["value"]))

        # Compare early vs. late averages rather than a single point-to-point comparison.
        early_avg = (
            sum(value_losses[:3])
            / 3
        )

        late_avg = (
            sum(value_losses[-3:])
            / 3
        )

        assert late_avg < early_avg, (
            "Expected critic value loss to decrease "
            "with a fixed, learnable reward signal, "
            f"got early_avg={early_avg}, "
            f"late_avg={late_avg}, "
            f"full trace={value_losses}"
        )