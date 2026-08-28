# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Distillation algorithm."""

from __future__ import annotations

import torch
import warnings
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 4
NUM_STEPS = 12
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_distillation_setup(
    gradient_length: int = 3,
    num_learning_epochs: int = 1,
    obs_normalization: bool = False,
    use_mixed_precision: bool = False,
    multi_gpu_cfg: dict | None = None,
) -> tuple:
    """Build a Distillation instance with small networks."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"student": ["policy"], "teacher": ["policy"]}

    student = MLPModel(
        obs, obs_groups, "student", NUM_ACTIONS, hidden_dims=[32, 32], obs_normalization=obs_normalization
    )
    teacher = MLPModel(obs, obs_groups, "teacher", NUM_ACTIONS, hidden_dims=[32, 32])

    storage = RolloutStorage("distillation", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])

    alg = Distillation(
        student,
        teacher,
        storage,
        num_learning_epochs=num_learning_epochs,
        gradient_length=gradient_length,
        learning_rate=1e-3,
        use_mixed_precision=use_mixed_precision,
        multi_gpu_cfg=multi_gpu_cfg,
    )
    return alg, obs, storage


def _fill_distillation_storage(alg: Distillation, obs: TensorDict) -> None:
    """Fill the distillation storage with transitions."""
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = alg.student(obs).detach()
        t.privileged_actions = alg.teacher(obs).detach()
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        alg.storage.add_transition(t)


class TestDistillationLoss:
    """Tests for distillation loss computation."""

    def test_loss_decreases_over_updates(self) -> None:
        """Behavior loss should decrease over repeated update() calls (learning signal works)."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3, num_learning_epochs=2)
        alg.train_mode()

        losses = []
        for _ in range(5):
            _fill_distillation_storage(alg, obs)
            loss_dict = alg.update()
            losses.append(loss_dict["behavior"])

        # Loss should generally decrease; allow some noise — check first vs last
        assert losses[-1] < losses[0], f"Loss should decrease over updates, got {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_gradient_accumulation_step_count(self) -> None:
        """Optimizer should step floor(num_transitions / gradient_length) times per epoch."""
        gradient_length = 4
        alg, obs, _storage = _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=1)
        alg.train_mode()

        _fill_distillation_storage(alg, obs)

        step_count = 0
        original_step = alg.optimizer.step

        def counting_step(*args: object, **kwargs: object) -> None:
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        alg.optimizer.step = counting_step
        alg.update()

        expected_steps = NUM_STEPS // gradient_length
        assert step_count == expected_steps, f"Expected {expected_steps} optimizer steps, got {step_count}"

    def test_update_changes_student_but_not_teacher(self) -> None:
        """Student parameters should change after update, while teacher parameters remain frozen."""
        alg, obs, _storage = _make_distillation_setup(gradient_length=3)
        alg.train_mode()

        student_before = {name: p.clone() for name, p in alg.student.named_parameters()}
        teacher_before = {name: p.clone() for name, p in alg.teacher.named_parameters()}

        _fill_distillation_storage(alg, obs)
        alg.update()

        any_student_changed = any(
            not torch.equal(p, student_before[name]) for name, p in alg.student.named_parameters()
        )
        assert any_student_changed, "Student parameters should change after an update"

        for name, p in alg.teacher.named_parameters():
            assert torch.equal(p, teacher_before[name]), f"Teacher parameter {name} changed during student update"

    def test_normalization_uses_rollout_after_student_update(self) -> None:
        """Student normalization should stay fixed during collection and update once from stored observations."""
        alg, _obs, storage = _make_distillation_setup(obs_normalization=True)

        for step in range(NUM_STEPS):
            rollout_obs = TensorDict(
                {"policy": torch.full((NUM_ENVS, OBS_DIM), float(step + 1))}, batch_size=[NUM_ENVS]
            )
            next_obs = TensorDict({"policy": torch.full((NUM_ENVS, OBS_DIM), float(step + 2))}, batch_size=[NUM_ENVS])
            alg.act(rollout_obs)
            alg.process_env_step(next_obs, torch.ones(NUM_ENVS), torch.zeros(NUM_ENVS), {})

        assert alg.student.obs_normalizer.count == 0

        expected_mean = storage.observations["policy"].flatten(0, 1).mean(dim=0)
        alg.update()

        assert alg.student.obs_normalizer.count == NUM_ENVS * NUM_STEPS
        assert torch.allclose(alg.student.obs_normalizer.mean, expected_mean)


class TestGradientBudgetWarning:
    """Tests for the construction-time check on the gradient accumulation budget."""

    def test_warns_when_budget_is_not_divisible(self) -> None:
        """A gradient_length of 5 leaves 2 of the 12 rollout steps in an accumulation that is never backpropagated."""
        with pytest.warns(UserWarning, match="The last 2 of 12 steps"):
            _make_distillation_setup(gradient_length=5)

    @pytest.mark.parametrize(
        ("gradient_length", "num_learning_epochs"),
        [(3, 1), (4, 1), (12, 1), (5, 5), (8, 2)],
    )
    def test_no_warning_when_budget_is_divisible(self, gradient_length: int, num_learning_epochs: int) -> None:
        """No warning is raised when the budget divides evenly by gradient_length.

        The budget is num_learning_epochs * num_transitions_per_env, so 5 epochs of 12 steps fit a gradient_length
        of 5 even though a single rollout does not.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_distillation_setup(gradient_length=gradient_length, num_learning_epochs=num_learning_epochs)
        assert not [w for w in caught if "gradient_length" in str(w.message)]


class TestDistillationMixedPrecision:
    """Tests for the use_mixed_precision flag in distillation."""

    def test_flag_defaults_to_false(self) -> None:
        """Mixed precision must be opt-in."""
        alg, _obs, _storage = _make_distillation_setup()
        assert alg.use_mixed_precision is False

    def test_update_runs_with_mixed_precision(self) -> None:
        """update() with the flag on returns a finite loss and changes student params."""
        alg, obs, _storage = _make_distillation_setup(
            gradient_length=3, num_learning_epochs=1, use_mixed_precision=True
        )
        alg.train_mode()
        _fill_distillation_storage(alg, obs)

        before = [p.clone() for p in alg.student.parameters()]
        loss_dict = alg.update()

        assert torch.isfinite(torch.tensor(loss_dict["behavior"]))
        after = list(alg.student.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after)), "student params should change"
