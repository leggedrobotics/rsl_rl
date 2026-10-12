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
   PyPI Package (Core Library) <https://pypi.org/project/rsl-rl-lib/>
   Main Documentation <https://leggedrobotics.github.io/rsl_rl/>

**RSL-RL** is a GPU-accelerated, lightweight learning library for robotics research. Its compact design allows
researchers to prototype and test new ideas without the overhead of modifying large, complex libraries. RSL-RL supports
multi-GPU training and features common algorithms for robot learning. The core library, without the additional features
of this branch, is also available via `PyPI <https://pypi.org/project/rsl-rl-lib/>`_.

Additional Features
-------------------

This is the documentation for the ``extras`` branch of RSL-RL, which contains additional features that are not part of
the core library. These features, mostly contributed by the community, are listed below and described in the
:ref:`overview <library-features>`.

- **Equivariant MLP model** that enforces robot symmetries (e.g. left-right mirroring) by construction.

Learning Environments
---------------------

RSL-RL is currently used by the following robot learning libraries:

- `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_ (built on top of NVIDIA Isaac Sim)
- `Legged Gym <https://github.com/leggedrobotics/legged_gym>`_ (built on top of NVIDIA Isaac Gym)
- `mjlab <https://github.com/mujocolab/mjlab>`_ (built on top of MuJoCo Warp)
- `MuJoCo Playground <https://github.com/google-deepmind/mujoco_playground>`_ (built on top of MuJoCo MJX and Warp)

Citation
--------

If you use RSL-RL in your research, please cite the `paper <https://arxiv.org/abs/2509.10771>`_:

.. code-block:: text

   @article{schwarke2025rslrl,
     title={RSL-RL: A Learning Library for Robotics Research},
     author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
     journal={arXiv preprint arXiv:2509.10771},
     year={2025}
   }

