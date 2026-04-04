"""Hydra-based Gymnasium evaluation entrypoint for rsl_rl examples."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from gymnasium_common import GymnasiumVecEnv, set_seed
from rsl_rl.runners import OffPolicyRunner, OnPolicyRunner


def to_plain_dict(node: DictConfig) -> dict[str, Any]:
    """Convert Hydra/OmegaConf nodes to plain Python dictionaries."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def build_train_cfg(cfg: DictConfig, train_cfg_key: str = "train_cfg") -> dict[str, Any]:
    """Build runner train_cfg directly from config without hardcoded hyperparameters."""
    train_cfg = deepcopy(to_plain_dict(cfg.algorithm[train_cfg_key]))
    train_cfg["run_name"] = cfg.algorithm.run_name
    train_cfg["logger"] = cfg.train.get("logger", "tensorboard")
    return train_cfg


def resolve_checkpoint(cfg: DictConfig) -> str:
    """Resolve checkpoint path from config, optionally auto-discovering latest model."""
    explicit = cfg.eval.checkpoint
    if explicit is not None:
        checkpoint = Path(str(explicit)).expanduser()
        if checkpoint.exists():
            return str(checkpoint)
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    log_dir = cfg.train.log_dir
    if log_dir is None:
        raise ValueError("No checkpoint specified and train.log_dir is null, cannot auto-discover checkpoint.")

    run_dir = Path(str(log_dir)).expanduser() / str(cfg.algorithm.run_name)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    candidates = list(run_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")

    def _extract_iter(path: Path) -> int:
        stem = path.stem
        suffix = stem.split("model_", maxsplit=1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    best = max(candidates, key=_extract_iter)
    return str(best)


def build_runner(cfg: DictConfig, env: GymnasiumVecEnv) -> OnPolicyRunner | OffPolicyRunner:
    """Construct the right runner for evaluation based on algorithm selection."""
    train_cfg = build_train_cfg(cfg)
    if str(cfg.algorithm.name) == "sac":
        return OffPolicyRunner(env=env, train_cfg=train_cfg, log_dir=None, device=cfg.device)
    return OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=None, device=cfg.device)


def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained policy on vectorized Gymnasium environments."""
    save_video = bool(cfg.eval.save_video)
    render_enabled = bool(cfg.eval.render) or save_video
    render_mode = "rgb_array" if save_video else (str(cfg.eval.render_mode) if render_enabled else None)

    if render_enabled and int(cfg.env.num_envs) != 1:
        raise ValueError("Rendering/video capture is only supported with env.num_envs=1 in eval mode.")

    if save_video and bool(cfg.eval.render) and str(cfg.eval.render_mode) != "rgb_array":
        raise ValueError("eval.save_video=true cannot be combined with eval.render_mode other than 'rgb_array'.")

    env = GymnasiumVecEnv(
        cfg.env.id,
        num_envs=cfg.env.num_envs,
        seed=cfg.seed,
        device=cfg.device,
        render_mode=render_mode,
    )
    checkpoint = resolve_checkpoint(cfg)

    runner = build_runner(cfg, env)
    runner.load(checkpoint, load_cfg=None, strict=True, map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)

    target_episodes = int(cfg.eval.num_episodes)
    max_steps = int(cfg.eval.max_steps_per_episode)
    stochastic_actions = bool(cfg.eval.stochastic_actions)
    print_episode_details = bool(cfg.eval.print_episode_details)

    video_writer = None
    video_path: Path | None = None
    if save_video:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError("imageio is required for eval.save_video=true. Install with `pip install imageio`.") from exc

        video_path = Path(str(cfg.eval.video_path)).expanduser()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=int(cfg.eval.video_fps))
        first_frame = env.render()
        if first_frame is not None:
            video_writer.append_data(first_frame)

    obs = env.get_observations().to(cfg.device)
    ep_returns = torch.zeros(env.num_envs, dtype=torch.float32, device=cfg.device)
    ep_lengths = torch.zeros(env.num_envs, dtype=torch.int64, device=cfg.device)
    finished_returns: list[float] = []
    finished_lengths: list[int] = []

    try:
        with torch.inference_mode():
            while len(finished_returns) < target_episodes:
                if stochastic_actions:
                    actions = policy(obs, stochastic_output=True)
                else:
                    actions = policy(obs)

                obs, rewards, dones, _extras = env.step(actions.to(env.device))
                obs = obs.to(cfg.device)
                rewards = rewards.to(cfg.device)
                dones = dones.to(cfg.device)

                if video_writer is not None:
                    frame = env.render()
                    if frame is not None:
                        video_writer.append_data(frame)

                ep_returns += rewards
                ep_lengths += 1

                finished_mask = (dones > 0.5) | (ep_lengths >= max_steps)
                done_indices = torch.nonzero(finished_mask).flatten().tolist()
                for idx in done_indices:
                    episode_return = float(ep_returns[idx].item())
                    episode_length = int(ep_lengths[idx].item())
                    finished_returns.append(episode_return)
                    finished_lengths.append(episode_length)
                    if print_episode_details:
                        ep_num = len(finished_returns)
                        print(f"Episode {ep_num:04d} | return={episode_return:.4f} | length={episode_length}")
                    ep_returns[idx] = 0.0
                    ep_lengths[idx] = 0
                    if len(finished_returns) >= target_episodes:
                        break
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    returns_t = torch.tensor(finished_returns[:target_episodes], dtype=torch.float32)
    lengths_t = torch.tensor(finished_lengths[:target_episodes], dtype=torch.float32)

    print("=== Evaluation Summary ===")
    print(f"Algorithm: {cfg.algorithm.name}")
    print(f"Environment: {cfg.env.id}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Episodes: {target_episodes}")
    print(f"Mean Return: {returns_t.mean().item():.4f}")
    print(f"Std Return:  {returns_t.std(unbiased=False).item():.4f}")
    print(f"Min Return:  {returns_t.min().item():.4f}")
    print(f"Max Return:  {returns_t.max().item():.4f}")
    print(f"Mean Length: {lengths_t.mean().item():.2f}")
    if video_path is not None:
        print(f"Video: {video_path}")


@hydra.main(version_base=None, config_path="configs", config_name="eval_gymnasium")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    evaluate(cfg)


if __name__ == "__main__":
    main()
