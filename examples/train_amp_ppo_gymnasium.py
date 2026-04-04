"""Train AMP-PPO on Gymnasium with optional expert observations dataset."""

from __future__ import annotations

import argparse

import torch

from gymnasium_common import GymnasiumVecEnv, build_amp_ppo_train_cfg, make_log_dir, set_seed
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AMP-PPO on Gymnasium with rsl_rl.")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps-per-env", type=int, default=64)
    parser.add_argument("--learning-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--expert-dataset", type=str, default=None)
    parser.add_argument("--expert-dataset-key", type=str, default="expert_observations")
    parser.add_argument("--expert-samples", type=int, default=8192)
    return parser.parse_args()


def collect_random_expert_observations(env: GymnasiumVecEnv, num_samples: int) -> torch.Tensor:
    """Collect a simple expert-observation buffer using random actions as a bootstrap dataset."""
    obs_buffer = []
    obs = env.get_observations()

    while len(obs_buffer) < num_samples:
        obs_buffer.append(obs["policy"].clone())
        random_actions = 2.0 * torch.rand(env.num_envs, env.num_actions, device=env.device) - 1.0
        obs, _rewards, _dones, _extras = env.step(random_actions)

    stacked = torch.cat(obs_buffer, dim=0)
    return stacked[:num_samples]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    env = GymnasiumVecEnv(args.env_id, num_envs=args.num_envs, seed=args.seed, device=args.device)

    expert_observations = None
    expert_path = args.expert_dataset
    if expert_path is None:
        expert_observations = collect_random_expert_observations(env, args.expert_samples)

    train_cfg = build_amp_ppo_train_cfg(
        num_steps_per_env=args.num_steps_per_env,
        expert_observations=expert_observations,
        expert_observations_path=expert_path,
        run_name="amp_ppo",
    )
    train_cfg["algorithm"]["expert_observations_key"] = args.expert_dataset_key

    log_dir = make_log_dir(args.log_dir, "amp_ppo")
    runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device=args.device)
    runner.learn(num_learning_iterations=args.learning_iterations)
    env.close()


if __name__ == "__main__":
    main()
