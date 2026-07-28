# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convergence example/test for MultiCriticPPO with separate critic observations.

This exercises the full act -> process_env_step -> compute_returns -> update loop
across varying critic counts and architectures.

Each critic has:
    - its own observation group
    - its own observation tensor
    - its own reward stream

For critic i:

    observation_i = critic_i
    reward_i = -0.02 * observation_i^2

Therefore each critic learns its own value function from its own observation
and reward stream.

The actor receives the "policy" observation group.

Configurations:
    - number of critics: 1, 2, 4, 5
    - architecture: small MLP, larger MLP, RNN (GRU)

Run as a pytest suite:
    pytest test_multi_critic_convergence_example.py -v

Run as a standalone demo:
    python test_multi_critic_convergence_example.py
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.runners import MultiCriticOnPolicyRunner
from rsl_rl.env import VecEnv


NUM_ENVS = 16
OBS_DIM = 6
NUM_ACTIONS = 3
MAX_EP_LEN = 1000
NUM_STEPS_PER_ENV = 16
DEVICE = "cpu"


class LearnableRewardEnv(VecEnv):
    """VecEnv with separate observations and rewards for each critic.

    The observation structure is:

        policy      -> actor observation
        critic_0    -> critic 0 observation
        critic_1    -> critic 1 observation
        ...
        critic_N    -> critic N observation

    Critic i learns:

        r_i(s_i) = -0.02 * s_i^2

    where s_i is the observation provided specifically to critic i.

    Thus each critic has:
        - a separate input observation
        - a separate reward stream
        - a separate value function target
    """

    def __init__(
        self,
        num_critics: int,
        device: str = DEVICE,
    ) -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.num_critics = num_critics
        self.max_episode_length = MAX_EP_LEN

        self.episode_length_buf = torch.zeros(
            NUM_ENVS,
            dtype=torch.long,
            device=device,
        )

        self.device = device
        self.cfg = {}

        # Keep the current observation so reward_i is computed from s_t.
        self.obs = self._sample_obs()

    def _sample_obs(self) -> TensorDict:
        """Generate a new observation for actor and every critic."""

        data = {
            # Actor gets its own observation.
            "policy": torch.randn(
                self.num_envs,
                OBS_DIM,
                device=self.device,
            ),
        }

        # Each critic gets a separate observation.
        #
        # critic_0 -> [num_envs, 1]
        # critic_1 -> [num_envs, 1]
        # ...
        for critic_idx in range(self.num_critics):
            data[f"critic_{critic_idx}"] = torch.randn(
                self.num_envs,
                1,
                device=self.device,
            )

        return TensorDict(
            data,
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        """Return the current observations."""
        return self.obs

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
        TensorDict,
        tuple[torch.Tensor, ...],
        torch.Tensor,
        dict,
    ]:
        """Advance the environment by one step."""

        # Current state s_t.
        current_obs = self.obs

        self.episode_length_buf += 1

        dones = (
            self.episode_length_buf >= self.max_episode_length
        ).float()

        self.episode_length_buf[dones.bool()] = 0

        # ---------------------------------------------------------
        # Each critic gets a reward based on ITS OWN observation.
        #
        # Critic 0:
        #   r_0 = -0.02 * critic_0^2
        #
        # Critic 1:
        #   r_1 = -0.02 * critic_1^2
        #
        # etc.
        # ---------------------------------------------------------
        rewards = tuple(
            -0.02
            * current_obs[f"critic_{critic_idx}"][:, 0].pow(2)
            for critic_idx in range(self.num_critics)
        )

        # Generate next state s_{t+1}.
        self.obs = self._sample_obs()

        extras = {
            "time_outs": torch.zeros(
                self.num_envs,
                device=self.device,
            )
        }

        return self.obs, rewards, dones, extras


def _make_cfg(model_type: str, num_critics: int) -> dict:
    cfg: dict = {
        "num_steps_per_env": NUM_STEPS_PER_ENV,

        "obs_groups": {
            "actor": ["policy"],
            **{
                f"critic_{i}": [f"critic_{i}"]
                for i in range(num_critics)
            },
        },

        "algorithm": {
            "class_name": "MultiCriticPPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "num_critics": num_critics,
            "learning_rate": 1e-3,
            "schedule": "fixed",
        },

        "multi_gpu": None,
    }

    if model_type == "rnn":
        cfg["actor"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }

        cfg["critic"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
        }

    elif model_type == "mlp_small":
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }

        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
        }

    else:
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }

        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
        }

    return cfg


def _train_and_collect_value_losses(
    model_type: str,
    num_critics: int,
    num_iterations: int,
) -> list[float]:
    """Run MultiCriticPPO training and collect mean value losses."""

    env = LearnableRewardEnv(
        num_critics=num_critics,
        device=DEVICE,
    )

    cfg = _make_cfg(
        model_type,
        num_critics,
    )

    runner = MultiCriticOnPolicyRunner(
        env=env,
        train_cfg=cfg,
        device=DEVICE,
    )

    loss_history = runner.learn(
        num_learning_iterations=num_iterations,
    )

    return [
        loss_dict["value"]
        for loss_dict in loss_history
    ]


class TestMultiCriticConvergenceAcrossConfigurations:
    """Value loss should decrease with separate critic observations."""

    NUM_ITERATIONS = 60

    CONVERGENCE_RATIO = 0.7

    @pytest.mark.parametrize(
        "num_critics",
        [1, 2, 4],
    )
    @pytest.mark.parametrize(
        "model_type",
        [
            "mlp_small",
            "mlp",
            "rnn",
        ],
    )
    def test_value_loss_decreases(
        self,
        model_type: str,
        num_critics: int,
    ) -> None:
        """Mean value loss should decrease during training."""

        torch.manual_seed(0)

        losses = _train_and_collect_value_losses(
            model_type,
            num_critics,
            self.NUM_ITERATIONS,
        )

        third = self.NUM_ITERATIONS // 3

        early_mean = (
            sum(losses[:third])
            / third
        )

        late_mean = (
            sum(losses[-third:])
            / third
        )

        assert late_mean < (
            early_mean
            * self.CONVERGENCE_RATIO
        ), (
            f"[{model_type}, "
            f"num_critics={num_critics}] "
            f"value loss did not converge: "
            f"early={early_mean:.4f}, "
            f"late={late_mean:.4f}"
        )

    def test_more_critics_do_not_break_convergence(
        self,
    ) -> None:
        """Five critics with separate observations should converge."""

        torch.manual_seed(0)

        losses = _train_and_collect_value_losses(
            "mlp",
            num_critics=5,
            num_iterations=self.NUM_ITERATIONS,
        )

        third = self.NUM_ITERATIONS // 3

        early_mean = (
            sum(losses[:third])
            / third
        )

        late_mean = (
            sum(losses[-third:])
            / third
        )

        assert late_mean < (
            early_mean
            * self.CONVERGENCE_RATIO
        ), (
            f"[mlp, num_critics=5] "
            f"value loss did not converge: "
            f"early={early_mean:.4f}, "
            f"late={late_mean:.4f}"
        )


if __name__ == "__main__":
    # Standalone demo.
    configs = [
        ("mlp_small", 1),
        ("mlp", 2),
        ("mlp", 4),
        ("mlp", 5),
        ("rnn", 2),
    ]

    for model_type, num_critics in configs:
        torch.manual_seed(0)

        print(
            f"\n=== model_type={model_type}, "
            f"num_critics={num_critics} ==="
        )

        losses = _train_and_collect_value_losses(
            model_type,
            num_critics,
            num_iterations=60,
        )

        for it, loss in enumerate(losses):
            if (
                it % 5 == 0
                or it == len(losses) - 1
            ):
                print(
                    f"  iter {it:3d} "
                    f"value_loss={loss:.4f}"
                )