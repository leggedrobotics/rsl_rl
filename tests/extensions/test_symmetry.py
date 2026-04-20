# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Symmetry extension."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.extensions.symmetry import Symmetry
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 6
NUM_ACTIONS = 3


def _negate_augmentation(
    env: object,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Toy data-augmentation callable: concatenate the originals with their element-wise negation.

    Produces a 2x-augmented batch ordered as ``[original, mirrored]`` along the batch dimension.
    """
    out_obs = None
    if obs is not None:
        neg = TensorDict({k: -v for k, v in obs.items()}, batch_size=obs.batch_size)
        out_obs = torch.cat([obs, neg], dim=0)

    out_actions = None
    if actions is not None:
        out_actions = torch.cat([actions, -actions], dim=0)

    return out_obs, out_actions


def _make_batch() -> RolloutStorage.Batch:
    """Create a minimal RL batch with matching shapes across all tensors."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    return RolloutStorage.Batch(
        observations=obs,
        actions=torch.randn(NUM_ENVS, NUM_ACTIONS),
        values=torch.randn(NUM_ENVS, 1),
        advantages=torch.randn(NUM_ENVS, 1),
        returns=torch.randn(NUM_ENVS, 1),
        old_actions_log_prob=torch.randn(NUM_ENVS, 1),
    )


def _make_actor() -> MLPModel:
    """Create a small MLP actor compatible with the observation dimensions used in this file."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"]}
    return MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


class TestAugmentBatch:
    """Tests for :meth:`Symmetry.augment_batch`."""

    def test_augmentation_doubles_batch(self) -> None:
        """Observations and actions should grow by ``num_aug`` along the batch dimension."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_data_augmentation=True)
        batch = _make_batch()

        sym.augment_batch(batch, NUM_ENVS)

        assert batch.observations.batch_size[0] == 2 * NUM_ENVS
        assert batch.actions.shape[0] == 2 * NUM_ENVS

    def test_scalar_tensors_are_repeated(self) -> None:
        """The remainingrollout tensors should be repeated ``num_aug`` times to match the augmented batch."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_data_augmentation=True)
        batch = _make_batch()

        orig_values = batch.values.clone()
        orig_advantages = batch.advantages.clone()
        orig_returns = batch.returns.clone()
        orig_log_prob = batch.old_actions_log_prob.clone()

        sym.augment_batch(batch, NUM_ENVS)

        for name, orig, new in [
            ("values", orig_values, batch.values),
            ("advantages", orig_advantages, batch.advantages),
            ("returns", orig_returns, batch.returns),
            ("old_actions_log_prob", orig_log_prob, batch.old_actions_log_prob),
        ]:
            assert new.shape[0] == 2 * NUM_ENVS, f"{name}: wrong batch size"
            assert torch.equal(new[:NUM_ENVS], orig), f"{name}: first slice should match original"
            assert torch.equal(new[NUM_ENVS:], orig), f"{name}: second slice should be a copy of the original"

    def test_observations_and_actions_are_mirrored(self) -> None:
        """The second half of the augmented batch should contain the mirrored samples, here: negated."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_data_augmentation=True)
        batch = _make_batch()

        orig_policy = batch.observations["policy"].clone()
        orig_actions = batch.actions.clone()

        sym.augment_batch(batch, NUM_ENVS)

        assert torch.allclose(batch.observations["policy"][:NUM_ENVS], orig_policy)
        assert torch.allclose(batch.observations["policy"][NUM_ENVS:], -orig_policy)
        assert torch.allclose(batch.actions[:NUM_ENVS], orig_actions)
        assert torch.allclose(batch.actions[NUM_ENVS:], -orig_actions)

    def test_noop_when_augmentation_disabled(self) -> None:
        """``augment_batch`` should be a no-op when :attr:`use_data_augmentation` is False."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_data_augmentation=False)
        batch = _make_batch()

        orig_obs_batch_size = batch.observations.batch_size
        orig_actions = batch.actions.clone()
        orig_values = batch.values.clone()

        sym.augment_batch(batch, NUM_ENVS)

        assert batch.observations.batch_size == orig_obs_batch_size
        assert torch.equal(batch.actions, orig_actions)
        assert torch.equal(batch.values, orig_values)


class TestComputeLoss:
    """Tests for :meth:`Symmetry.compute_loss`."""

    def test_returns_finite_scalar(self) -> None:
        """The mirror MSE should be a finite, non-negative scalar."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_mirror_loss=True)
        actor = _make_actor()
        batch = _make_batch()

        loss = sym.compute_loss(actor, batch, NUM_ENVS)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_graph_connected_when_mirror_loss_enabled(self) -> None:
        """The returned loss must require gradients when the mirror loss contributes to the policy update."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_mirror_loss=True)
        actor = _make_actor()
        batch = _make_batch()

        loss = sym.compute_loss(actor, batch, NUM_ENVS)

        assert loss.requires_grad

    def test_detached_when_mirror_loss_disabled(self) -> None:
        """When the mirror loss is off the loss is still returned for logging, but detached from the graph."""
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_mirror_loss=False)
        actor = _make_actor()
        batch = _make_batch()

        loss = sym.compute_loss(actor, batch, NUM_ENVS)

        assert not loss.requires_grad

    def test_augments_observations_when_not_previously_augmented(self) -> None:
        """If ``augment_batch`` wasn't called, ``compute_loss`` augments the observations in place."""
        sym = Symmetry(
            env=None,
            data_augmentation_func=_negate_augmentation,
            use_data_augmentation=False,
            use_mirror_loss=True,
        )
        actor = _make_actor()
        batch = _make_batch()

        assert batch.observations.batch_size[0] == NUM_ENVS
        sym.compute_loss(actor, batch, NUM_ENVS)
        assert batch.observations.batch_size[0] == 2 * NUM_ENVS

    def test_does_not_re_augment_when_batch_already_augmented(self) -> None:
        """When data augmentation already happened upstream, ``compute_loss`` must not augment again."""
        sym = Symmetry(
            env=None,
            data_augmentation_func=_negate_augmentation,
            use_data_augmentation=True,
            use_mirror_loss=True,
        )
        actor = _make_actor()
        batch = _make_batch()

        sym.augment_batch(batch, NUM_ENVS)
        assert batch.observations.batch_size[0] == 2 * NUM_ENVS

        sym.compute_loss(actor, batch, NUM_ENVS)
        assert batch.observations.batch_size[0] == 2 * NUM_ENVS

    def test_zero_loss_when_actor_is_symmetry_equivariant(self) -> None:
        """If the actor respects the symmetry (here: odd function), the mirror loss should be (near) zero.

        The toy augmentation negates observations and actions. An actor whose action mean is an odd function of the
        observation satisfies ``f(-x) == -f(x)``, so the MSE between ``f(-x)`` and ``-f(x)`` should vanish.
        """
        sym = Symmetry(env=None, data_augmentation_func=_negate_augmentation, use_mirror_loss=True)
        batch = _make_batch()

        # Replace the actor with a deterministic odd function: f(x) = W x (no bias)
        weight = torch.randn(NUM_ACTIONS, OBS_DIM)

        def odd_actor(obs: TensorDict) -> torch.Tensor:
            return obs["policy"] @ weight.t()

        loss = sym.compute_loss(odd_actor, batch, NUM_ENVS)  # type: ignore[arg-type]
        assert loss.item() < 1e-6
