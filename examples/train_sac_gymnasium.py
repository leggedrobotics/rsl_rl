"""Train SAC on a Gymnasium continuous classic benchmark using rsl_rl."""

from __future__ import annotations

from gymnasium_common import GymnasiumVecEnv, build_sac_train_cfg, make_log_dir, parse_common_args, set_seed
from rsl_rl.runners import OffPolicyRunner


def main() -> None:
    args = parse_common_args("Train SAC on Gymnasium with rsl_rl.")
    set_seed(args.seed)

    env = GymnasiumVecEnv(
        env_id=args.env_id,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
    )
    log_dir = make_log_dir(args.log_dir, "sac")
    train_cfg = build_sac_train_cfg(args.num_steps_per_env, run_name="sac")

    runner = OffPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device=args.device)
    runner.learn(num_learning_iterations=args.learning_iterations)
    env.close()


if __name__ == "__main__":
    main()
