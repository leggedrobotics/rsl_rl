Configuration
=============

RSL-RL is configured with a dictionary that is passed to RSL-RL's runner during initialization. The dictionary is
usually read from a YAML file or constructed from python dataclasses, such as in Isaac Lab. The dictionary is nested to 
reflect the structure of the library, and follows the following pattern:

.. figure:: ../_static/rsl_rl_config_light.png
   :width: 100%
   :align: center
   :class: light-only


.. figure:: ../_static/rsl_rl_config_dark.png
   :width: 100%
   :align: center
   :class: dark-only

The following sections list the available settings for each configuration component. Most settings are optional and get 
assigned default values if not specified.

Runner Configuration
--------------------

Currently, RSL-RL supports two runner classes:
:class:`~rsl_rl.runners.on_policy_runner.OnPolicyRunner` and
:class:`~rsl_rl.runners.distillation_runner.DistillationRunner`. The 
:class:`~rsl_rl.runners.on_policy_runner.OnPolicyRunner` is configured as follows:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``num_steps_per_env``
     - int
     - required
     - Number of environment steps collected per iteration.
   * - ``obs_groups``
     - dict[str, list[str]]
     - required
     - Mapping from observation sets to observation tensors coming from the environment.
   * - ``run_name``
     - string
     - missing
     - Optional run label shown in the console output.
   * - ``save_interval``
     - int
     - required
     - Number of iterations between checkpoints.
   * - ``logger``
     - string
     - ``tensorboard``
     - Logging service to use. Valid values: ``tensorboard``, ``wandb``, ``neptune``.
   * - ``wandb_project``
     - string
     - required for W&B
     - W&B project name used by the W&B writer.
   * - ``neptune_project``
     - string
     - required for Neptune
     - Neptune project name used by the Neptune writer.
   * - ``algorithm``
     - dict
     - required
     - RL algorithm configuration.
   * - ``actor``
     - dict
     - required
     - Actor model configuration.
   * - ``critic``
     - dict
     - required
     - Critic model configuration.


For the :class:`~rsl_rl.runners.distillation_runner.DistillationRunner`, the ``actor`` and ``critic`` keys are simply
replaced by ``student`` and ``teacher`` keys, respectively:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ...
     - ...
     - ...
     - ...
   * - ``student``
     - dict
     - required
     - Student model configuration.
   * - ``teacher``
     - dict
     - required
     - Teacher model configuration.

Algorithm Configuration
-----------------------

RSL-RL implements two algorithms, :class:`~rsl_rl.algorithms.ppo.PPO` and 
:class:`~rsl_rl.algorithms.distillation.Distillation`. The :class:`~rsl_rl.algorithms.ppo.PPO` algorithm is configured
as follows:

.. list-table::
   :header-rows: 1

   * - Key
     - Type
     - Default
     - Description
   * - ``class_name``
     - str
     - ``PPO``
     - Algorithm class name.
   * - ``optimizer``
     - str
     - ``"adam"``
     - Optimizer used for policy/value updates. Valid values: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"rmsprop"``.
   * - ``learning_rate``
     - float
     - ``0.001``
     - Optimizer learning rate.
   * - ``num_learning_epochs``
     - int
     - ``5``
     - Number of optimization epochs per iteration.
   * - ``num_mini_batches``
     - int
     - ``4``
     - Number of mini-batches used in each optimization epoch.
   * - ``schedule``
     - str
     - ``"adaptive"``
     - Learning-rate schedule. Valid values: ``"adaptive"``, ``"fixed"``.
   * - ``value_loss_coef``
     - float
     - ``1.0``
     - Coefficient for value-function loss.
   * - ``clip_param``
     - float
     - ``0.2``
     - PPO clipping parameter for surrogate/value clipping.
   * - ``use_clipped_value_loss``
     - bool
     - ``True``
     - Whether to use clipped value loss.
   * - ``desired_kl``
     - float
     - ``0.01``
     - Target KL divergence used by the adaptive schedule.
   * - ``entropy_coef``
     - float
     - ``0.01``
     - Entropy regularization coefficient.
   * - ``gamma``
     - float
     - ``0.99``
     - Discount factor.
   * - ``lam``
     - float
     - ``0.95``
     - GAE lambda parameter.
   * - ``max_grad_norm``
     - float
     - ``1.0``
     - Gradient clipping norm.
   * - ``normalize_advantage_per_mini_batch``
     - bool
     - ``False``
     - Whether to normalize advantages separately for each mini-batch.
   * - ``rnd_cfg``
     - dict | None
     - ``None``
     - Optional Random Network Distillation configuration.
   * - ``symmetry_cfg``
     - dict | None
     - ``None``
     - Optional symmetry augmentation/loss configuration.
