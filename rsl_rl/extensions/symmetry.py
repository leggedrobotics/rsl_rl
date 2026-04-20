# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable


class Symmetry:
    """Symmetry data augmentation and mirror loss.

    The extension supports two (optionally simultaneous) uses of a user-provided symmetry function:

    - :attr:`use_data_augmentation` appends mirrored observation/action pairs to every mini-batch, so that the policy
      and value loss are evaluated on both the original and the mirrored samples.
    - :attr:`use_mirror_loss` adds an auxiliary MSE term that penalizes the policy for disagreeing with itself when
      evaluated on mirrored observations.

    If both flags are disabled the symmetry loss is still computed for logging purposes but detached from the graph.

    References:
        - Mittal et al. "Symmetry Considerations for Learning Task Symmetric Robot Policies." ICRA (2024).
    """

    def __init__(
        self,
        env: VecEnv,
        data_augmentation_func: str | Callable,
        use_data_augmentation: bool = False,
        use_mirror_loss: bool = False,
        mirror_loss_coeff: float = 0.0,
    ) -> None:
        """Initialize the symmetry extension.

        Args:
            env: Environment object. Passed to the data augmentation function for handling different observation terms.
            data_augmentation_func: Callable that generates mirrored observations / actions. Resolved using
                :func:`~rsl_rl.utils.utils.resolve_callable`.
            use_data_augmentation: Whether to append mirrored samples to every mini-batch.
            use_mirror_loss: Whether to add an auxiliary mirror loss term to the loss function.
            mirror_loss_coeff: Scaling factor applied to the mirror loss when :attr:`use_mirror_loss` is True.
        """
        # Symmetry parameters
        self.env = env
        self.use_data_augmentation = use_data_augmentation
        self.use_mirror_loss = use_mirror_loss
        self.mirror_loss_coeff = mirror_loss_coeff

        # Resolve the augmentation function
        self.data_augmentation_func = resolve_callable(data_augmentation_func)

        # Inform the user if symmetry is configured only for logging
        if not (use_data_augmentation or use_mirror_loss):
            print("Symmetry not used for learning. We will use it for logging instead.")

    def augment_batch(self, batch: RolloutStorage.Batch, original_batch_size: int) -> None:
        """Augment the mini-batch in place with mirrored observations and actions.

        After the call ``batch.observations`` and ``batch.actions`` have shape ``[original_batch_size * num_aug, ...]``
        with the original samples in the first slice and the mirrored samples in the remaining slices. The remaining
        rollout tensors (old log probabilities, values, advantages, returns) are repeated to match.

        When :attr:`use_data_augmentation` is False, the batch is left unchanged.
        """
        if not self.use_data_augmentation:
            return
        # Returned shape: [original_batch_size * num_aug, ...]
        batch.observations, batch.actions = self.data_augmentation_func(
            env=self.env,
            obs=batch.observations,
            actions=batch.actions,
        )
        # Repeat the remaining rollout tensors to match the augmented observations/actions
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

    def compute_loss(self, actor: MLPModel, batch: RolloutStorage.Batch, original_batch_size: int) -> torch.Tensor:
        """Compute the mirror loss between the actor's action means on original and mirrored observations.

        If :meth:`augment_batch` has not been called for this batch (i.e. :attr:`use_data_augmentation` is False), the
        observations are augmented here first so that the actor is evaluated on both the original and the mirrored
        samples.

        The returned loss is detached when :attr:`use_mirror_loss` is False so that it can be reported for logging
        without contributing to gradients.
        """
        # Augment observations if the batch has not already been augmented
        if not self.use_data_augmentation:
            batch.observations, _ = self.data_augmentation_func(env=self.env, obs=batch.observations, actions=None)

        # Action means predicted by the actor on the augmented observation batch
        mean_actions = actor(batch.observations.detach().clone())

        # Mirror the original-slice action means using the augmentation function. We use the action means here rather
        # than the sampled actions in ``batch.actions``, since the symmetry loss is defined on the policy mean.
        _, mean_actions_symm = self.data_augmentation_func(
            env=self.env, obs=None, actions=mean_actions[:original_batch_size]
        )

        # MSE between the actor prediction on mirrored obs and the mirrored actor prediction on the original obs
        symmetry_loss = nn.functional.mse_loss(
            mean_actions[original_batch_size:],
            mean_actions_symm.detach()[original_batch_size:],
        )
        return symmetry_loss if self.use_mirror_loss else symmetry_loss.detach()


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the symmetry configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # If using symmetry then pass the environment object
    # Note: This is used by the symmetry function for handling different observation terms
    if "symmetry_cfg" in alg_cfg and alg_cfg["symmetry_cfg"] is not None:
        alg_cfg["symmetry_cfg"]["env"] = env
    else:
        alg_cfg["symmetry_cfg"] = None
    return alg_cfg
