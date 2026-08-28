# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distributed training tests for PPO and Distillation."""

from __future__ import annotations

import datetime
import os
import tempfile
import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from itertools import chain
from tensordict import TensorDict
from typing import Any

import pytest

import rsl_rl.algorithms.distillation as distillation_mod
import rsl_rl.algorithms.ppo as ppo_mod
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage
from tests.algorithms.test_distillation import _fill_distillation_storage, _make_distillation_setup
from tests.algorithms.test_ppo import NUM_ENVS, NUM_STEPS, OBS_DIM, _build_ppo
from tests.conftest import make_obs

_GLOO_AVAILABLE = torch.distributed.is_available() and torch.distributed.is_gloo_available()

# Small enough that the test models mix packed buckets and oversized slices.
_TINY_BUCKET_BYTES = 256


def spawn_gloo(worker: Callable[..., None], *extra_args: Any, world_size: int = 2) -> None:
    """Run ``worker(rank, world_size, store_file, *extra_args)`` under a gloo process group."""
    with tempfile.TemporaryDirectory() as tmp:
        store_file = os.path.join(tmp, "filestore")
        torch.multiprocessing.spawn(
            worker,
            args=(world_size, store_file, *extra_args),
            nprocs=world_size,
            join=True,
        )


def init_gloo(rank: int, world_size: int, store_file: str) -> None:
    """Initialize a gloo process group using a FileStore."""
    store = torch.distributed.FileStore(store_file, world_size)
    torch.distributed.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )


def _assert_close_across_ranks(tensor: torch.Tensor) -> None:
    """Gather ``tensor`` and assert every rank holds the same values."""
    gathered = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    for other in gathered[1:]:
        assert torch.allclose(gathered[0], other)


def _assert_not_close_across_ranks(tensor: torch.Tensor) -> None:
    """Gather ``tensor`` and assert ranks hold different values."""
    gathered = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    assert not torch.allclose(gathered[0], gathered[1])


def _clone_params_with_grads(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    """Clone parameters and their gradients for a reference reduction."""
    clones = []
    for param in params:
        clone = nn.Parameter(param.detach().clone())
        clone.grad = None if param.grad is None else param.grad.clone()
        clones.append(clone)
    return clones


def _reduce_with_full_concat(parameters: Iterable[nn.Parameter], world_size: int) -> None:
    """Concatenate every gradient into one buffer and average it across ranks."""
    params = list(parameters)
    grads = [param.grad.view(-1) for param in params if param.grad is not None]
    packed = torch.cat(grads)
    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    packed /= world_size
    offset = 0
    for param in params:
        if param.grad is not None:
            numel = param.numel()
            param.grad.copy_(packed[offset : offset + numel].view_as(param.grad))
            offset += numel


def _assert_param_grads_equal(left: Iterable[nn.Parameter], right: Iterable[nn.Parameter]) -> None:
    """Assert two parameter lists hold matching gradients, including ``None``."""
    for left_param, right_param in zip(left, right):
        if left_param.grad is None:
            assert right_param.grad is None
        else:
            assert torch.equal(left_param.grad, right_param.grad)


def _assign_mixed_grads(params: list[nn.Parameter], rank: int) -> None:
    """Leave the first parameter without a grad and fill the rest deterministically."""
    torch.manual_seed(rank + 1)
    for i, param in enumerate(params):
        param.grad = None if i == 0 else torch.randn_like(param)


def _fill_ppo_storage(ppo: PPO, obs: TensorDict) -> None:
    """Fill PPO rollout storage with a short on-policy trajectory."""
    for _ in range(NUM_STEPS):
        t = RolloutStorage.Transition()
        t.observations = obs
        t.hidden_states = (None, None)
        t.actions = ppo.actor(obs, stochastic_output=True).detach()
        t.values = ppo.critic(obs).detach()
        t.actions_log_prob = ppo.actor.get_output_log_prob(t.actions).detach()
        t.distribution_params = tuple(p.detach() for p in ppo.actor.output_distribution_params)
        t.rewards = torch.randn(NUM_ENVS)
        t.dones = torch.zeros(NUM_ENVS)
        ppo.storage.add_transition(t)


def _reduce_compare_worker(rank: int, world_size: int, store_file: str) -> None:
    init_gloo(rank, world_size, store_file)
    try:
        ppo_mod._GRAD_REDUCE_BUCKET_BYTES = _TINY_BUCKET_BYTES
        distillation_mod._GRAD_REDUCE_BUCKET_BYTES = _TINY_BUCKET_BYTES

        torch.manual_seed(0)
        ppo, _obs = _build_ppo(multi_gpu_cfg={"global_rank": rank, "world_size": world_size})
        ppo_params = list(chain(ppo.actor.parameters(), ppo.critic.parameters()))
        _assign_mixed_grads(ppo_params, rank)
        grad_bytes = [param.numel() * param.element_size() for param in ppo_params if param.grad is not None]
        assert any(size > _TINY_BUCKET_BYTES for size in grad_bytes)
        assert any(size <= _TINY_BUCKET_BYTES for size in grad_bytes)
        ppo_ref = _clone_params_with_grads(ppo_params)
        ppo.reduce_parameters()
        _reduce_with_full_concat(ppo_ref, world_size)
        _assert_param_grads_equal(ppo_params, ppo_ref)
        assert ppo_params[0].grad is None

        torch.manual_seed(0)
        alg, _obs, _storage = _make_distillation_setup(multi_gpu_cfg={"global_rank": rank, "world_size": world_size})
        student_params = list(alg.student.parameters())
        _assign_mixed_grads(student_params, rank + 2)
        student_ref = _clone_params_with_grads(student_params)
        alg.reduce_parameters()
        _reduce_with_full_concat(student_ref, world_size)
        _assert_param_grads_equal(student_params, student_ref)
        assert student_params[0].grad is None
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _ppo_worker(rank: int, world_size: int, store_file: str) -> None:
    init_gloo(rank, world_size, store_file)
    try:
        torch.manual_seed(0)
        ppo, obs = _build_ppo(
            schedule="fixed",
            multi_gpu_cfg={"global_rank": rank, "world_size": world_size},
        )
        ppo.broadcast_parameters()
        ppo.train_mode()

        torch.manual_seed(rank + 7)
        _fill_ppo_storage(ppo, obs)
        ppo.compute_returns(obs)
        ppo.update()

        _assert_close_across_ranks(next(ppo.actor.parameters()).detach())
        _assert_close_across_ranks(next(ppo.critic.parameters()).detach())
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _distillation_worker(rank: int, world_size: int, store_file: str, reduce_grads: bool = True) -> None:
    init_gloo(rank, world_size, store_file)
    try:
        torch.manual_seed(0)
        alg, _obs, _storage = _make_distillation_setup(
            gradient_length=3,
            num_learning_epochs=1,
            multi_gpu_cfg={"global_rank": rank, "world_size": world_size},
        )
        alg.broadcast_parameters()
        alg.train_mode()
        if not reduce_grads:
            alg.reduce_parameters = lambda: None  # type: ignore[method-assign]

        # Behavior cloning depends on observations and teacher actions, not rewards.
        torch.manual_seed(rank + 11)
        _fill_distillation_storage(alg, make_obs(NUM_ENVS, OBS_DIM))
        alg.update()

        student_param = next(alg.student.parameters()).detach()
        if reduce_grads:
            _assert_close_across_ranks(student_param)
        else:
            _assert_not_close_across_ranks(student_param)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class TestReduceParametersLocal:
    """Single-process tests for ``reduce_parameters``."""

    def test_none_gradients_are_left_untouched(self) -> None:
        """Parameters without gradients must stay ``None`` and must not require dist init."""
        ppo, _obs = _build_ppo(multi_gpu_cfg={"global_rank": 0, "world_size": 2})
        params = list(chain(ppo.actor.parameters(), ppo.critic.parameters()))
        for param in params:
            param.grad = None
        ppo.reduce_parameters()
        for param in params:
            assert param.grad is None


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason="torch.distributed gloo backend is required")
class TestDistributedAlgorithms:
    """Multi-GPU tests for gradient reduction and on-policy updates."""

    def test_chunked_reduce_matches_full_concat(self) -> None:
        """Bucketed reduction must match the previous full-buffer average, including oversized and ``None`` grads."""
        spawn_gloo(_reduce_compare_worker)

    def test_ppo_parameters_stay_in_sync(self) -> None:
        """PPO ranks that see different rollouts must still share parameters after update."""
        spawn_gloo(_ppo_worker)

    def test_distillation_student_stays_in_sync(self) -> None:
        """Distillation ranks must keep the student parameters aligned after update."""
        spawn_gloo(_distillation_worker)

    def test_distillation_student_diverges_without_reduction(self) -> None:
        """Different per-rank observations must change the student if gradients are not averaged."""
        spawn_gloo(_distillation_worker, False)
