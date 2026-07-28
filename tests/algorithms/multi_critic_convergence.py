# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convergence example/test for MultiCriticPPO across varying critic counts and architectures.

This exercises the full act -> process_env_step -> compute_returns -> update loop (the same core
loop `OnPolicyRunner.learn()` runs, minus logging) across a matrix of configurations:
    - number of critics: 1, 2, 4, 5
    - critic/actor architecture: small MLP, larger MLP, RNN (GRU)

The environment returns i.i.d. observations with a reward that is a deterministic function of the
*current* observation, independent of the action taken. This gives a well-defined regression target
for the critic(s): since observations are i.i.d., the true value function is
    V(s) = r(s) + gamma * E[V]
i.e. a bounded, learnable function of the current state. That makes "does the value loss go down"
a meaningful, fast-to-check convergence signal, regardless of how many critics are used or what
architecture backs them.

Run as a pytest suite:
    pytest test_multi_critic_convergence_example.py -v

Run as a standalone demo (prints per-iteration value loss curves):
    python test_multi_critic_convergence_example.py
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MultiCriticPPO
from rsl_rl.env import VecEnv

NUM_ENVS = 16
OBS_DIM = 6
NUM_ACTIONS = 3
MAX_EP_LEN = 1000  # long enough that termination essentially never happens during a short run
NUM_STEPS_PER_ENV = 16
DEVICE = "cpu"


class LearnableRewardEnv(VecEnv):
    """VecEnv with i.i.d. observations and a reward that is a deterministic function of the
    observation, so the critic has a well-defined, learnable regression target and value loss
    should decrease steadily as training progresses.
    """

    def __init__(self, device: str = DEVICE) -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}

    def _sample_obs(self) -> TensorDict:
        data = {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)}
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:  # noqa: D102
        return self._sample_obs()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self._sample_obs()
        # Deterministic, bounded reward derived purely from the *current* observation (independent
        # of the action), so the critic's regression target is well-defined. Kept small (0.02, not
        # 0.1) so the resulting value target (~ reward / (1 - gamma)) stays in an easy range for
        # low-capacity critics to regress onto within a short training run.
        rewards = -0.02 * (obs["policy"] ** 2).sum(dim=-1)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_cfg(model_type: str, num_critics: int) -> dict:
    """Build a minimal MultiCriticPPO config for the given actor/critic architecture.

    Args:
        model_type: One of ``"mlp_small"``, ``"mlp"``, or ``"rnn"``.
        num_critics: Number of critics for MultiCriticPPO to construct.
    """
    cfg: dict = {
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "MultiCriticPPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "num_critics": num_critics,
            "learning_rate": 1e-3,
            "schedule": "fixed",  # keep the learning rate fixed for a cleaner convergence signal
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
            "distribution_cfg": {"class_name": "GaussianDistribution"},
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
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [16],
            "activation": "elu",
        }
    else:  # "mlp"
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [64, 64],
            "activation": "elu",
        }
    return cfg


def _train_and_collect_value_losses(model_type: str, num_critics: int, num_iterations: int) -> list[float]:
    """Build a MultiCriticPPO algorithm and run a short training loop, returning the per-iteration
    mean value loss (averaged across all critics, exactly as `MultiCriticPPO.update()` reports it).

    Mirrors the rollout/update loop in `OnPolicyRunner.learn()` (minus logging), so we can collect
    the loss history directly instead of going through the `Logger`.
    """
    env = LearnableRewardEnv()
    cfg = _make_cfg(model_type, num_critics)
    obs = env.get_observations()

    alg = MultiCriticPPO.construct_algorithm(obs, env, cfg, DEVICE)
    alg.train_mode()

    value_losses: list[float] = []
    for _ in range(num_iterations):
        with torch.inference_mode():
            for _ in range(NUM_STEPS_PER_ENV):
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions)
                alg.process_env_step(obs, rewards, dones, extras)
            alg.compute_returns(obs)
        loss_dict = alg.update()
        value_losses.append(loss_dict["value"])

    return value_losses


class TestMultiCriticConvergenceAcrossConfigurations:
    """Value loss should trend downward for a range of critic counts and architectures."""

    NUM_ITERATIONS = 60
    # Convergence threshold: late-training value loss must drop below this fraction of early-training
    # value loss. Kept at 0.7 (rather than something tighter like 0.5) so the check isn't overtuned to
    # any one architecture/capacity combination — the point is verifying real, sustained improvement,
    # not chasing a specific numeric target.
    CONVERGENCE_RATIO = 0.7

    @pytest.mark.parametrize("num_critics", [1, 2, 4])
    @pytest.mark.parametrize("model_type", ["mlp_small", "mlp", "rnn"])
    def test_value_loss_decreases(self, model_type: str, num_critics: int) -> None:
        """Mean value loss over the final third of training should be well below the first third."""
        torch.manual_seed(0)
        losses = _train_and_collect_value_losses(model_type, num_critics, self.NUM_ITERATIONS)

        third = self.NUM_ITERATIONS // 3
        early_mean = sum(losses[:third]) / third
        late_mean = sum(losses[-third:]) / third

        assert late_mean < early_mean * self.CONVERGENCE_RATIO, (
            f"[{model_type}, num_critics={num_critics}] value loss did not converge: "
            f"early={early_mean:.4f}, late={late_mean:.4f}"
        )

    def test_more_critics_do_not_break_convergence(self) -> None:
        """A larger critic ensemble (5 critics) should still converge, just like a single critic."""
        torch.manual_seed(0)
        losses = _train_and_collect_value_losses("mlp", num_critics=5, num_iterations=self.NUM_ITERATIONS)
        third = self.NUM_ITERATIONS // 3
        early_mean = sum(losses[:third]) / third
        late_mean = sum(losses[-third:]) / third
        assert late_mean < early_mean * self.CONVERGENCE_RATIO, (
            f"[mlp, num_critics=5] value loss did not converge: early={early_mean:.4f}, late={late_mean:.4f}"
        )


if __name__ == "__main__":
    # Standalone demo: prints the value-loss convergence curve for each configuration so you can
    # visually confirm that adding critics / changing architecture doesn't break learning.
    configs = [
        ("mlp_small", 1),
        ("mlp", 2),
        ("mlp", 4),
        ("mlp", 5),
        ("rnn", 2),
    ]
    for model_type, num_critics in configs:
        torch.manual_seed(0)
        print(f"\n=== model_type={model_type}, num_critics={num_critics} ===")
        losses = _train_and_collect_value_losses(model_type, num_critics, num_iterations=60)
        for it, loss in enumerate(losses):
            if it % 5 == 0 or it == len(losses) - 1:
                print(f"  iter {it:3d}  value_loss={loss:.4f}")