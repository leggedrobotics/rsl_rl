# RSL RL

Fast and simple implementation of RL algorithms, designed to run fully on GPU.

Currently, the following algorithms are implemented:
- Distributed Distributional DDPG (D4PG)
- Deep Deterministic Policy Gradient (DDPG)
- Distributional PPO (DPPO)
- Distributional Soft Actor Critic (DSAC)
- Proximal Policy Optimization (PPO)
- Soft Actor Critic (SAC)
- Twin Delayed DDPG (TD3)

**Maintainer**: David Hoeller, Nikita Rudin <br/>
**Affiliation**: Robotic Systems Lab, ETH Zurich & NVIDIA <br/>
**Contact**: Nikita Rudin (rudinn@ethz.ch), Lukas Schneider (lukas@luschneider.com)

## Citation

If you use our code in your research, please cite us:
```
@misc{schneider2023learning,
  archivePrefix={arXiv},
  author={Lukas Schneider and Jonas Frey and Takahiro Miki and Marco Hutter},
  eprint={2309.14246},
  primaryClass={cs.RO}
  title={Learning Risk-Aware Quadrupedal Locomotion using Distributional Reinforcement Learning}, 
  year={2023},
}
```

## Installation

To install the package, run the following command in the root directory of the repository:

```bash
$ pip3 install -e .
```

Examples can be run from the `examples/` directory.
The example directory also include hyperparameters tuned for some gym environments.
These are automatically loaded when running the example.
Videos of the trained policies are periodically saved to the `videos/` directory.

```bash
$ python3 examples/example.py
```

To run gym mujoco environments, you need a working installation of the mujoco simulator and [mujoco_py](https://github.com/openai/mujoco-py).

## Tests

The repository contains a set of tests to ensure that the algorithms are working as expected.
To run the tests, simply execute:

```bash
$ cd tests/ && python -m unittest
```

## Documentation

To generate documentation, run the following command in the root directory of the repository:

```bash
$ pip3 install sphinx sphinx-rtd-theme
$ sphinx-apidoc -o docs/source . ./examples
$ cd docs/ && make html
```

## Contribution Guidelines

We use [`black`](https://github.com/psf/black) formatter for formatting the python code.
You should [configure `black` with VSCode](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0) or you can manually format files with:

```bash
$ pip install black
$ black --line-length 120 .
```
