# RSL-RL

**RSL-RL** is a GPU-accelerated, lightweight learning library for robotics research. Its compact design allows
researchers to prototype and test new ideas without the overhead of modifying large, complex libraries. RSL-RL can also
be used out-of-the-box by installing it via [PyPI](https://pypi.org/project/rsl-rl-lib/), supports multi-GPU training,
and features common algorithms for robot learning.

## Key Features

- **Minimal, readable codebase** with clear extension points for rapid prototyping.
- **Robotics-first methods** including PPO and Student-Teacher Distillation.
- **High-throughput training** with native Multi-GPU support.
- **Proven performance** in numerous research publications.

## Learning Environments

RSL-RL is currently used by the following robot learning libraries:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (built on top of NVIDIA Isaac Sim)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym) (built on top of NVIDIA Isaac Gym)
- [mjlab](https://github.com/mujocolab/mjlab) (built on top of MuJoCo Warp)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (built on top of MuJoCo MJX and Warp)

## Installation

Before installing RSL-RL, ensure that Python `3.9+` is available. It is recommended to install the library in a virtual
environment (e.g. using `venv` or `conda`), which is often already created by the used environment library (e.g.
Isaac Lab). If so, make sure to activate it before installing RSL-RL.

### Installing RSL-RL as a dependency

```bash
pip install rsl-rl-lib
```

### Installing RSL-RL for development

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

## Citation

If you use RSL-RL in your research, please cite the [paper](https://arxiv.org/abs/2509.10771):

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```

## Version matrix (Chrono / Gymnasium integrations)

| `rsl-rl-lib` | Config schema | Notes |
|--------------|---------------|-------|
| **2.3.x** (e.g. 2.3.3) | Top-level `policy` with `class_name: ActorCritic`; flat `algorithm` dict | Used by [gym-chrono](https://github.com/projectchrono/gym-chrono) Chrono 10 reproducers and [Chrono::Ray](https://github.com/uwsbel/chrono-ray) quadruped-style scripts. Example: `config/examples/chrono_gymnasium_ppo_2x.yaml`. |
| **5.x / main** | `actor` + `critic` `MLPModel` blocks; required `obs_groups`; `TensorDict` observations | See [configuration guide](docs/guide/configuration.rst). Example: `config/examples/chrono_gymnasium_ppo_current.yaml`. |

When upgrading `rsl-rl-lib`, diff your YAML against the table above — field renames are **not** backward compatible between the 2.x ActorCritic layout and the current MLPModel layout.

## PyTorch on AMD ROCm

RSL-RL uses generic `torch` ops (no custom CUDA extensions in the default PPO path). Training on **AMD Instinct** GPUs works when you install **PyTorch built for ROCm** and pass `device="cuda"` to `OnPolicyRunner` (PyTorch uses the HIP/ROCm backend).

```bash
pip install rsl-rl-lib==2.3.3   # or your pinned minor
pip install --force-reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.2
```

Use a ROCm wheel index that matches your driver (`rocm6.2`, `rocm6.3`, etc.). After installing RSL-RL, re-pin ROCm `torch` if pip resolves a CUDA build.

**Related:** [gym-chrono](https://github.com/projectchrono/gym-chrono) README (version pins, Ray + ROCm layout); [chrono-ray](https://github.com/uwsbel/chrono-ray) `docs/AMD_ROCM.md` for Ray bootstrap before `import ray`.
