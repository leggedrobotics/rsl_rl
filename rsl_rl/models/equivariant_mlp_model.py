# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.equivariant import EquivariantMLP, SignedPermutation

__all__ = ["EquivariantMLPModel"]


class EquivariantMLPModel(MLPModel):
    r"""MLP model whose network is strictly equivariant under a symmetry of the robot.

    Many legged platforms are left-right symmetric. Exploiting that symmetry through data augmentation or a
    mirror loss makes the policy *approximately* equivariant: both are soft and can be traded away against
    the task reward. This model instead constrains the network itself, so

    .. math::
        \pi(M s) = M \pi(s), \qquad V(M s) = V(s)

    hold by construction at every step of training. Mittal et al., *Leveraging Symmetry in RL-based Legged
    Locomotion Control* (IROS 2024), report that a strictly equivariant network outperforms augmentation in
    sample efficiency, task performance, gait quality and zero-shot transfer.

    The model only replaces the MLP head of :class:`~rsl_rl.models.MLPModel`; observation selection,
    normalization, distribution handling, export and hidden-state management are inherited unchanged.

    .. note::
        Observation normalization is not compatible with equivariance, because per-dimension statistics are
        generally not symmetric. Passing ``obs_normalization=True`` therefore raises.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
    ) -> None:
        """Initialize the equivariant MLP model.

        Args:
            obs: Observation dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g. "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP. Each must be even, since the hidden layers carry the
                regular representation of the order-two group.
            activation: Activation function of the MLP.
            obs_normalization: Must be False; normalization would break equivariance.
            distribution_cfg: Configuration dictionary for the output distribution. To keep *sampling*
                equivariant and not only the mean, use
                :class:`~rsl_rl.modules.EquivariantGaussianDistribution`.
            symmetry_cfg: Dictionary with the symmetry representations::

                    {
                        "obs":    {"perm": [...], "sign": [...]},   # acts on the concatenated observation
                        "output": {"perm": [...], "sign": [...]},   # acts on the output; omit to make the
                                                                    # model invariant, as for a critic
                    }

        Raises:
            ValueError: If ``symmetry_cfg`` is missing, if ``obs_normalization`` is requested, or if the
                observation representation does not match the observation dimension.
        """
        if symmetry_cfg is None or "obs" not in symmetry_cfg:
            raise ValueError("EquivariantMLPModel requires symmetry_cfg with an 'obs' representation")
        if obs_normalization:
            raise ValueError(
                "obs_normalization breaks equivariance: the running statistics are not symmetric. "
                "Disable it, or normalize with symmetry-aware statistics outside the model."
            )

        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )

        rep_in = SignedPermutation(**symmetry_cfg["obs"])
        if len(rep_in) != self._get_latent_dim():
            raise ValueError(
                f"symmetry_cfg['obs'] acts on {len(rep_in)} dimensions but the '{obs_set}' observation is"
                f" {self._get_latent_dim()}-dimensional"
            )

        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        if "output" in symmetry_cfg:
            rep_out = SignedPermutation(**symmetry_cfg["output"])
            if len(rep_out) != mlp_output_dim:
                raise ValueError(
                    f"symmetry_cfg['output'] acts on {len(rep_out)} dimensions but the MLP output is"
                    f" {mlp_output_dim}-dimensional. Note that a stochastic model may widen the output."
                )
        else:
            # No output representation: the model is invariant, which is what a critic needs.
            rep_out = SignedPermutation.identity(mlp_output_dim)

        # Replace the MLP head. Everything else in MLPModel is reused as is.
        self.mlp = EquivariantMLP(rep_in, rep_out, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)
