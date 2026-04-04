# Gymnasium Examples

This folder contains end-to-end examples showing how to use rsl_rl algorithms on Gymnasium classic continuous-control benchmarks.

The examples now use a single Hydra entrypoint and config groups to select the training algorithm.

## Requirements

- Install rsl_rl in editable mode:

```bash
pip install -e .
```

- Install Gymnasium + Hydra:

```bash
pip install gymnasium[classic-control] hydra-core
```

- Install TensorBoard (used by default logging setup):

```bash
pip install tensorboard
```

## Unified Entrypoint

Use one script for all Gymnasium examples:

```bash
python examples/train_gymnasium.py algorithm=ppo
```

Evaluate a trained checkpoint:

```bash
python examples/eval_gymnasium.py algorithm=ppo eval.checkpoint=./logs/ppo/model_10000.pt
```

Algorithm config files live in `examples/configs/algorithm`:

- `ppo.yaml`
- `sac.yaml`
- `amp_ppo.yaml`
- `dagger_ppo.yaml`
- `distillation.yaml`

## Quick Start

Run PPO on Pendulum:

```bash
python examples/train_gymnasium.py algorithm=ppo env.id=Pendulum-v1 device=cpu
```

Run PPO on MountainCarContinuous:

```bash
python examples/train_gymnasium.py algorithm=ppo env.id=MountainCarContinuous-v0 device=cpu
```

Run SAC on Pendulum:

```bash
python examples/train_gymnasium.py algorithm=sac env.id=Pendulum-v1 device=cpu
```

Run DaggerPPO (with auto teacher pre-training):

```bash
python examples/train_gymnasium.py algorithm=dagger_ppo env.id=Pendulum-v1 device=cpu
```

Run Distillation (with auto teacher pre-training):

```bash
python examples/train_gymnasium.py algorithm=distillation env.id=Pendulum-v1 device=cpu
```

Run AMP-PPO:

```bash
python examples/train_gymnasium.py algorithm=amp_ppo env.id=Pendulum-v1 device=cpu
```

## Common Overrides

- `env.num_envs=16`: parallel environments
- `train.base_run_name=ppo`: base name for run folders
- `train.run_name=${train.base_run_name}_${now:%Y%m%d_%H%M%S}`: default run folder naming (timestamped)
- `algorithm.train_cfg.num_steps_per_env=64`: rollout steps per update
- `train.learning_iterations=60`: number of training iterations
- `algorithm.teacher.learning_iterations=40`: teacher pre-training iterations (DaggerPPO / Distillation)
- `train.log_dir=./logs`: enable TensorBoard logging and checkpoints
- `train.log_dir=null`: disable file logging and console training summaries
- `train.logger=tensorboard`: logging backend (`tensorboard`, `wandb`, `neptune`)
- `teacher.checkpoint=/path/to/checkpoint.pt`: load an existing teacher checkpoint
- `algorithm.amp.expert_dataset=/path/to/expert.pt`: load expert observations for AMP-PPO

Most training hyperparameters are now in `algorithm.train_cfg` (and `algorithm.teacher.train_cfg` when applicable).
You can create a new config file and override via Hydra defaults, or override ad-hoc from CLI.
Hydra run artifacts now default directly under the active run directory (for example `./logs/ppo_20260404_070000/hydra/` containing `.hydra/` and `train_gymnasium.log`) instead of project-root `outputs/...`.

## Evaluation Overrides

- `eval.checkpoint=/path/to/model.pt`: explicit checkpoint path
- `eval.checkpoint=null`: auto-pick latest `model_*.pt` from `train.log_dir/algorithm.run_name`
- `eval.num_episodes=20`: number of episodes used for metrics
- `eval.max_steps_per_episode=1000`: hard cap for episode horizon in evaluation loop
- `eval.stochastic_actions=false`: use deterministic actions (`true` samples stochastic policy actions)
- `eval.print_episode_details=true`: print per-episode return/length during rollout
- `eval.render=true`: enable Gymnasium rendering (requires `env.num_envs=1`)
- `eval.render_mode=human`: Gymnasium render mode passed to environment creation
- `eval.save_video=true`: save an MP4 rollout video (requires `env.num_envs=1`)
- `eval.rendering_subdir=rendering`: output subdirectory under checkpoint run dir
- `eval.exported_subdir=exported`: ONNX subdirectory under checkpoint run dir
- `eval.video_path=null`: auto path `<run_dir>/rendering/ckpt_<iter>/<timestamp>.mp4`
- `eval.video_fps=30`: FPS used for MP4 encoding
- `eval.export_onnx=true`: export ONNX policy by default during evaluation
- `eval.onnx_path=null`: auto path `<run_dir>/exported/ckpt_<iter>/policy.onnx`
