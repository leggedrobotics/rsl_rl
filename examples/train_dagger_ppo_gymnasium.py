"""Train DaggerPPO on Gymnasium with a PPO teacher checkpoint."""

from __future__ import annotations

import argparse
import os

from gymnasium_common import GymnasiumVecEnv, build_ppo_train_cfg, make_log_dir, set_seed
from rsl_rl.runners import OnPolicyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DaggerPPO on Gymnasium with rsl_rl.")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps-per-env", type=int, default=64)
    parser.add_argument("--teacher-iters", type=int, default=40)
    parser.add_argument("--learning-iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    return parser.parse_args()


def build_dagger_cfg(num_steps_per_env: int, run_name: str | None = None) -> dict:
    cfg = {
        "run_name": run_name,
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 1000,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"], "teacher": ["policy"]},
        "algorithm": {
            "class_name": "DaggerPPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.0,
            "learning_rate": 3e-4,
            "max_grad_norm": 1.0,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "dagger_loss_coef": 1.0,
            "loss_type": "mse",
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
        },
        "teacher": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
    }
    return cfg


def train_teacher_if_needed(args: argparse.Namespace) -> str:
    if args.teacher_checkpoint is not None:
        return args.teacher_checkpoint

    env = GymnasiumVecEnv(args.env_id, num_envs=args.num_envs, seed=args.seed, device=args.device)
    teacher_cfg = build_ppo_train_cfg(args.num_steps_per_env, run_name="teacher_ppo")
    teacher_log_dir = make_log_dir(args.log_dir, "teacher_ppo")
    teacher_runner = OnPolicyRunner(env=env, train_cfg=teacher_cfg, log_dir=teacher_log_dir, device=args.device)
    teacher_runner.learn(num_learning_iterations=args.teacher_iters)

    if teacher_log_dir is None:
        checkpoint_path = os.path.join(".", "teacher_ppo_checkpoint.pt")
    else:
        checkpoint_path = os.path.join(teacher_log_dir, "teacher_ppo_checkpoint.pt")
    teacher_runner.save(checkpoint_path)
    env.close()
    return checkpoint_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    teacher_checkpoint = train_teacher_if_needed(args)

    env = GymnasiumVecEnv(args.env_id, num_envs=args.num_envs, seed=args.seed + 1000, device=args.device)
    dagger_cfg = build_dagger_cfg(args.num_steps_per_env, run_name="dagger_ppo")
    dagger_log_dir = make_log_dir(args.log_dir, "dagger_ppo")
    dagger_runner = OnPolicyRunner(env=env, train_cfg=dagger_cfg, log_dir=dagger_log_dir, device=args.device)

    dagger_runner.load(teacher_checkpoint, load_cfg=None, strict=True, map_location=args.device)
    dagger_runner.learn(num_learning_iterations=args.learning_iterations)
    env.close()


if __name__ == "__main__":
    main()
