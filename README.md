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

Before installing, ensure that Python `3.9+` is available. It is recommended to install the library in a virtual
environment (e.g. using `venv` or `conda`), which is often already created by the used environment library (e.g.
`Isaac Lab`). If so, make sure to activate it before installing RSL-RL.

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

If you use RSL-RL in your research, please cite the following paper:

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```
