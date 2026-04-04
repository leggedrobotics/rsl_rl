"""Hydra-based unified Gymnasium training entrypoint for rsl_rl examples."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from gymnasium_common import GymnasiumVecEnv, make_log_dir, set_seed
from rsl_rl.runners import DistillationRunner, OffPolicyRunner, OnPolicyRunner


def to_plain_dict(node: DictConfig) -> dict[str, Any]:
    """Convert Hydra/OmegaConf nodes to plain Python dictionaries."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def build_train_cfg(cfg: DictConfig, train_cfg_key: str = "train_cfg") -> dict:
    """Build runner train_cfg directly from config without hardcoded hyperparameters."""
    train_cfg = deepcopy(to_plain_dict(cfg.algorithm[train_cfg_key]))
    train_cfg["run_name"] = cfg.algorithm.run_name
    train_cfg["logger"] = cfg.train.get("logger", "tensorboard")
    return train_cfg


def collect_random_expert_observations(env: GymnasiumVecEnv, num_samples: int) -> torch.Tensor:
    """Collect a bootstrap expert-observation buffer with random actions."""
    obs_buffer = []
    obs = env.get_observations()

    while len(obs_buffer) < num_samples:
        obs_buffer.append(obs["policy"].clone())
        random_actions = 2.0 * torch.rand(env.num_envs, env.num_actions, device=env.device) - 1.0
        obs, _rewards, _dones, _extras = env.step(random_actions)

    stacked = torch.cat(obs_buffer, dim=0)
    return stacked[:num_samples]


def train_teacher_if_needed(cfg: DictConfig) -> str:
    """Return teacher checkpoint path, training a PPO teacher if needed."""
    checkpoint = cfg.teacher.checkpoint
    if checkpoint is not None:
        return str(checkpoint)

    env = GymnasiumVecEnv(
        env_id=cfg.env.id,
        num_envs=cfg.env.num_envs,
        seed=cfg.seed,
        device=cfg.device,
    )
    teacher_cfg = deepcopy(to_plain_dict(cfg.algorithm.teacher.train_cfg))
    teacher_cfg["run_name"] = cfg.algorithm.teacher.run_name
    teacher_cfg["logger"] = cfg.train.get("logger", "tensorboard")
    teacher_log_dir = make_log_dir(cfg.train.log_dir, cfg.algorithm.teacher.run_name)
    teacher_runner = OnPolicyRunner(env=env, train_cfg=teacher_cfg, log_dir=teacher_log_dir, device=cfg.device)
    teacher_runner.learn(num_learning_iterations=cfg.algorithm.teacher.learning_iterations)

    if teacher_log_dir is None:
        checkpoint_path = cfg.algorithm.teacher.checkpoint_name
    else:
        checkpoint_path = f"{teacher_log_dir}/{cfg.algorithm.teacher.checkpoint_name}"
    teacher_runner.save(checkpoint_path)
    env.close()
    return checkpoint_path


def run_ppo(cfg: DictConfig) -> None:
    env = GymnasiumVecEnv(cfg.env.id, num_envs=cfg.env.num_envs, seed=cfg.seed, device=cfg.device)
    train_cfg = build_train_cfg(cfg)
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=make_log_dir(cfg.train.log_dir, cfg.algorithm.run_name),
        device=cfg.device,
    )
    runner.learn(num_learning_iterations=cfg.train.learning_iterations)
    env.close()


def run_sac(cfg: DictConfig) -> None:
    env = GymnasiumVecEnv(cfg.env.id, num_envs=cfg.env.num_envs, seed=cfg.seed, device=cfg.device)
    train_cfg = build_train_cfg(cfg)
    runner = OffPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=make_log_dir(cfg.train.log_dir, cfg.algorithm.run_name),
        device=cfg.device,
    )
    runner.learn(num_learning_iterations=cfg.train.learning_iterations)
    env.close()


def run_amp_ppo(cfg: DictConfig) -> None:
    env = GymnasiumVecEnv(cfg.env.id, num_envs=cfg.env.num_envs, seed=cfg.seed, device=cfg.device)

    expert_observations = None
    expert_path = cfg.algorithm.amp.expert_dataset
    if expert_path is None:
        expert_observations = collect_random_expert_observations(env, int(cfg.algorithm.amp.expert_samples))

    train_cfg = build_train_cfg(cfg)
    train_cfg["algorithm"]["expert_observations_key"] = cfg.algorithm.amp.expert_dataset_key
    if expert_path is None:
        train_cfg["algorithm"]["expert_observations"] = expert_observations
    else:
        train_cfg["algorithm"]["expert_observations_path"] = expert_path

    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=make_log_dir(cfg.train.log_dir, cfg.algorithm.run_name),
        device=cfg.device,
    )
    runner.learn(num_learning_iterations=cfg.train.learning_iterations)
    env.close()


def run_dagger_ppo(cfg: DictConfig) -> None:
    teacher_checkpoint = train_teacher_if_needed(cfg)

    env = GymnasiumVecEnv(
        cfg.env.id,
        num_envs=cfg.env.num_envs,
        seed=cfg.seed + int(cfg.algorithm.seed_offset),
        device=cfg.device,
    )
    train_cfg = build_train_cfg(cfg)
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=make_log_dir(cfg.train.log_dir, cfg.algorithm.run_name),
        device=cfg.device,
    )

    runner.load(teacher_checkpoint, load_cfg=None, strict=True, map_location=cfg.device)
    runner.learn(num_learning_iterations=cfg.train.learning_iterations)
    env.close()


def run_distillation(cfg: DictConfig) -> None:
    teacher_checkpoint = train_teacher_if_needed(cfg)

    env = GymnasiumVecEnv(
        cfg.env.id,
        num_envs=cfg.env.num_envs,
        seed=cfg.seed + int(cfg.algorithm.seed_offset),
        device=cfg.device,
    )
    train_cfg = build_train_cfg(cfg)
    runner = DistillationRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=make_log_dir(cfg.train.log_dir, cfg.algorithm.run_name),
        device=cfg.device,
    )

    runner.load(teacher_checkpoint, load_cfg=None, strict=True, map_location=cfg.device)
    runner.learn(num_learning_iterations=cfg.train.learning_iterations)
    env.close()


@hydra.main(version_base=None, config_path="configs", config_name="train_gymnasium")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    algorithm_name = str(cfg.algorithm.name)
    run_map = {
        "ppo": run_ppo,
        "sac": run_sac,
        "amp_ppo": run_amp_ppo,
        "dagger_ppo": run_dagger_ppo,
        "distillation": run_distillation,
    }

    if algorithm_name not in run_map:
        supported = ", ".join(sorted(run_map.keys()))
        raise ValueError(f"Unsupported algorithm '{algorithm_name}'. Supported: {supported}")

    run_map[algorithm_name](cfg)


if __name__ == "__main__":
    main()
