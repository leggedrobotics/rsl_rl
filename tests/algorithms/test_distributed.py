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

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets
from tests.algorithms.test_distillation import _fill_distillation_storage, _make_distillation_setup
from tests.algorithms.test_ppo import NUM_ENVS, NUM_STEPS, OBS_DIM, _build_ppo
from tests.conftest import make_obs

_GLOO_AVAILABLE = torch.distributed.is_available() and torch.distributed.is_gloo_available()

# Small enough that the test models mix packed buckets and oversized slices.
_TINY_BUCKET_BYTES = 256
_TINY_BUCKET_MB = _TINY_BUCKET_BYTES / (1024 * 1024)


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
        torch.manual_seed(0)
        ppo, _obs = _build_ppo(multi_gpu_cfg={"global_rank": rank, "world_size": world_size})
        ppo_params = list(chain(ppo.actor.parameters(), ppo.critic.parameters()))
        _assign_mixed_grads(ppo_params, rank)
        grad_bytes = [param.numel() * param.element_size() for param in ppo_params if param.grad is not None]
        assert any(size > _TINY_BUCKET_BYTES for size in grad_bytes)
        assert any(size <= _TINY_BUCKET_BYTES for size in grad_bytes)
        ppo_ref = _clone_params_with_grads(ppo_params)
        reduce_gradients_in_buckets(ppo_params, world_size, bucket_mb=_TINY_BUCKET_MB)
        _reduce_with_full_concat(ppo_ref, world_size)
        _assert_param_grads_equal(ppo_params, ppo_ref)
        assert ppo_params[0].grad is None

        torch.manual_seed(0)
        alg, _obs, _storage = _make_distillation_setup(multi_gpu_cfg={"global_rank": rank, "world_size": world_size})
        student_params = list(alg.student.parameters())
        _assign_mixed_grads(student_params, rank + 2)
        student_ref = _clone_params_with_grads(student_params)
        reduce_gradients_in_buckets(student_params, world_size, bucket_mb=_TINY_BUCKET_MB)
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


def _sac_worker(rank: int, world_size: int, store_file: str, categorical: bool) -> None:
    from rsl_rl.algorithms import SAC
    from tests.algorithms.test_sac import _make_train_cfg
    from tests.runners.test_on_policy_runner import DummyEnv

    init_gloo(rank, world_size, store_file)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(rank + 10)
        cfg = _make_train_cfg()
        cfg["multi_gpu"] = {"global_rank": rank, "world_size": world_size}
        cfg["algorithm"].update(
            num_updates=3,
            actor_update_interval=2,
            reward_normalization=True,
            critic_loss_type="categorical" if categorical else "mse",
            num_bins=11,
            max_grad_norm=0.1,
            grad_reduce_bucket_mb=_TINY_BUCKET_MB,
            learning_rate_schedule="cosine",
            num_schedule_updates=20,
        )
        for role in ("actor", "critic"):
            cfg[role]["network_cfg"] = {"class_name": "ResidualMLP", "hidden_dim": 16, "num_blocks": 1, "expansion": 2}
        env = DummyEnv()
        obs = env.get_observations()
        alg = SAC.construct_algorithm(obs, env, cfg, "cpu")

        def assert_synchronized(learner: SAC) -> None:
            for model in (learner.actor, learner.critics, learner.target_critics, learner._return_normalizer):
                for value in model.state_dict().values():
                    assert torch.isfinite(value).all()
                    _assert_close_across_ranks(value)
            _assert_close_across_ranks(learner.log_alpha.detach())
            _assert_close_across_ranks(learner._max_abs_return)
            for optimizer in (learner.actor_optimizer, learner.critic_optimizer, learner.alpha_optimizer):
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            _assert_close_across_ranks(value)

        assert_synchronized(alg)
        reduced = []
        reduce_parameters = alg.reduce_parameters

        def check_reduction(optimizer: torch.optim.Optimizer) -> None:
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            expected = _clone_params_with_grads(parameters)
            _reduce_with_full_concat(expected, world_size)
            reduce_parameters(optimizer)
            for parameter, reference in zip(parameters, expected):
                if reference.grad is not None:
                    torch.testing.assert_close(parameter.grad, reference.grad)
            reduced.append(optimizer)

        alg.reduce_parameters = check_reduction
        obs["policy"] = torch.full_like(obs["policy"], float(rank * 4))
        with torch.inference_mode():
            for _ in range(2):
                alg.act(obs)
                alg.process_env_step(obs, torch.full((4,), float(rank + 1)), torch.zeros(4), {})
        losses = alg.update()
        assert losses and all(torch.isfinite(torch.tensor(value)) for value in losses.values())
        assert {id(opt) for opt in reduced} == {
            id(alg.actor_optimizer),
            id(alg.critic_optimizer),
            id(alg.alpha_optimizer),
        }
        assert alg.num_critic_updates == 3
        assert_synchronized(alg)
        torch.testing.assert_close(alg.actor.obs_normalizer.mean, torch.full((8,), 2.0))
        torch.testing.assert_close(alg.actor.obs_normalizer._var, torch.full((1, 8), 4.0))
        assert alg.actor.obs_normalizer.count == 16
        returns = torch.tensor([1.0, 1.99, 2.0, 3.98])
        torch.testing.assert_close(alg._return_normalizer.mean, returns.mean().reshape(1))
        torch.testing.assert_close(alg._return_normalizer._var, returns.var(unbiased=False).reshape(1, 1))
        torch.testing.assert_close(alg._max_abs_return, returns.max())
        _assert_not_close_across_ranks(alg.storage.observations["policy"][0])

        old_steps = alg.storage.num_steps
        if rank == 0:
            alg.storage.num_steps = 0
        assert alg.update() == {}
        assert alg.num_critic_updates == 3
        alg.storage.num_steps = old_steps

        import io

        checkpoint = io.BytesIO()
        torch.save(alg.save(), checkpoint)
        checkpoint.seek(0)
        restored = SAC.construct_algorithm(obs, env, cfg, "cpu")
        restored.load(torch.load(checkpoint, weights_only=False), None, True)
        restored.broadcast_parameters()
        assert restored.num_critic_updates == 3
        assert restored._actor_loaded
        assert restored._schedulers[0].last_epoch == 3
        assert_synchronized(restored)
        restored.act(obs)
        restored.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
        assert restored.update()
        assert restored.num_critic_updates == 6
        assert_synchronized(restored)

        final = obs.clone()
        final["policy"].fill_(float("nan"))
        dones = torch.full((4,), rank == 1)
        restored.act(obs)
        before = restored.storage.size
        with pytest.raises(ValueError, match="nonfinite final_obs"):
            restored.process_env_step(obs, torch.ones(4), dones, {"final_obs": final})
        assert restored.storage.size == before
        cfg["algorithm"]["use_cuda_graph"] = True
        with pytest.raises(ValueError, match="Distributed SAC does not support CUDA graph"):
            SAC.construct_algorithm(obs, env, cfg, "cpu")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not _GLOO_AVAILABLE, reason="torch.distributed gloo backend is required")
@pytest.mark.parametrize("categorical", [False, True])
def test_sac_distributed_updates_and_resume(categorical: bool) -> None:
    """Different local replay data must preserve globally synchronized learner state and normalization."""
    spawn_gloo(_sac_worker, categorical)
