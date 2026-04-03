# Gymnasium Examples

This folder contains end-to-end examples showing how to use rsl_rl algorithms on Gymnasium classic continuous-control benchmarks.

The examples use a lightweight adapter that exposes Gymnasium environments through the rsl_rl VecEnv interface.

## Requirements

- Install rsl_rl in editable mode:

```bash
pip install -e .
```

- Install Gymnasium:

```bash
pip install gymnasium[classic-control]
```

## Included Examples

- `train_ppo_gymnasium.py`
  - PPO on a Gymnasium environment (default: `Pendulum-v1`).

- `train_sac_gymnasium.py`
  - SAC on a Gymnasium environment (default: `Pendulum-v1`).

- `train_dagger_ppo_gymnasium.py`
  - DaggerPPO with a PPO teacher.
  - If `--teacher-checkpoint` is not provided, a teacher is pre-trained automatically.

- `train_distillation_gymnasium.py`
  - Distillation runner with a PPO teacher.
  - If `--teacher-checkpoint` is not provided, a teacher is pre-trained automatically.

## Quick Start

Run PPO on Pendulum:

```bash
python examples/train_ppo_gymnasium.py --env-id Pendulum-v1 --device cpu
```

Run PPO on MountainCarContinuous:

```bash
python examples/train_ppo_gymnasium.py --env-id MountainCarContinuous-v0 --device cpu
```

Run SAC on Pendulum:

```bash
python examples/train_sac_gymnasium.py --env-id Pendulum-v1 --device cpu
```

Run DaggerPPO (with auto teacher pre-training):

```bash
python examples/train_dagger_ppo_gymnasium.py --env-id Pendulum-v1 --device cpu
```

Run Distillation (with auto teacher pre-training):

```bash
python examples/train_distillation_gymnasium.py --env-id Pendulum-v1 --device cpu
```

## Useful Flags

- `--num-envs`: parallel environments (default: 16)
- `--num-steps-per-env`: rollout steps per update (default: 64)
- `--learning-iterations`: number of training iterations
- `--teacher-iters`: teacher pre-training iterations (for DaggerPPO / Distillation)
- `--log-dir`: enable TensorBoard logging and checkpoints under this directory
- `--teacher-checkpoint`: load an existing teacher checkpoint
