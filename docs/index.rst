RSL-RL Documentation
====================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Guide
   
   guide/overview
   guide/installation
   guide/configuration
   guide/contribution

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference
   
   api/algorithms
   api/env
   api/extensions
   api/models
   api/modules
   api/runners
   api/storage
   api/utils

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project Links

   GitHub Repository <https://github.com/leggedrobotics/rsl_rl>
   PyPI Package <https://pypi.org/project/rsl-rl-lib/>

**RSL-RL** is a GPU-accelerated, lightweight learning library for robotics research. It's compact design allows 
researchers to prototype and test new ideas without the overhead of modifying large, complex libraries. RSL-RL can also 
be used out-of-the-box by installing it via `PyPI <https://pypi.org/project/rsl-rl-lib/>`_, supports multi-GPU training 
and features common algorithms for robot learning.

Key Features
------------

- **Minimal, readable codebase** with clear extension points for rapid prototyping.
- **Robotics-first methods** including PPO and Student-Teacher Distillation.
- **High-throughput training** with native Multi-GPU support.
- **Proven performance** in numerous research publications.

Learning Environments
---------------------

RSL-RL is currently used by the following robot learning libraries:

- `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_ (built on top of NVIDIA Isaac Sim)
- `Legged Gym <https://github.com/leggedrobotics/legged_gym>`_ (built on top of NVIDIA Isaac Gym)
- `mjlab <https://github.com/mujocolab/mjlab>`_ (built on top of MuJoCo Warp)
- `MuJoCo Playground <https://github.com/google-deepmind/mujoco_playground>`_ (built on top of MuJoCo MJX and Warp)

Citation
--------

If you use RSL-RL in your research, please cite the following paper:

.. code-block:: text

   @article{schwarke2025rslrl,
     title={RSL-RL: A Learning Library for Robotics Research},
     author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
     journal={arXiv preprint arXiv:2509.10771},
     year={2025}
   }

