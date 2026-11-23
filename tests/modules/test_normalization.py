# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for normalization modules."""

import torch

import pytest

from rsl_rl.modules.normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from tests.algorithms.test_distributed import init_gloo, spawn_gloo


def _double_moments_worker(rank: int, world_size: int, store_file: str) -> None:
    """Compare unequal distributed batches and repeated updates with pooled double-precision samples."""
    init_gloo(rank, world_size, store_file)
    try:
        normalizer = EmpiricalNormalization(1).double()
        pooled = []
        for step in range(3):
            batches = [
                (torch.arange(worker + step + 1, dtype=torch.float64) + worker * 7 + step * 2).unsqueeze(-1)
                for worker in range(world_size)
            ]
            normalizer.update(batches[rank])
            pooled.extend(batches)
            reference = torch.cat(pooled)
            torch.testing.assert_close(normalizer.mean, reference.mean(0), rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(normalizer.std, reference.std(0, unbiased=False), rtol=1e-13, atol=1e-13)
            expected = (batches[rank] - reference.mean(0)) / (reference.std(0, unbiased=False) + normalizer.eps)
            torch.testing.assert_close(normalizer(batches[rank]), expected, rtol=1e-13, atol=1e-13)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class TestEmpiricalNormalization:
    """Tests for ``EmpiricalNormalization``."""

    @pytest.mark.parametrize("dtype,value", [(torch.float32, 2e38), (torch.float64, 1e308)])
    def test_large_finite_constant_batches_remain_normalizable(self, dtype: torch.dtype, value: float) -> None:
        """Finite constant observations retain zero variance from the first batch through later updates."""
        normalizer = EmpiricalNormalization(2).to(dtype=dtype)
        for batch_size in (8, 3, 1):
            observations = torch.full((batch_size, 2), value, dtype=dtype)
            normalizer.update(observations)
            torch.testing.assert_close(normalizer.mean, observations[0], rtol=0.0, atol=0.0)
            torch.testing.assert_close(normalizer.std, torch.zeros(2, dtype=dtype), rtol=0.0, atol=0.0)
            torch.testing.assert_close(normalizer(observations), torch.zeros_like(observations), rtol=0.0, atol=0.0)
        assert normalizer.count == 12

    @pytest.mark.parametrize("dtype,value", [(torch.float32, 3e19), (torch.float64, 2e154)])
    def test_large_batch_mean_changes_preserve_representable_variance(self, dtype: torch.dtype, value: float) -> None:
        """Weighting the between-batch correction must precede an otherwise overflowing product."""
        normalizer = EmpiricalNormalization(1).to(dtype=dtype)
        observations = torch.tensor([[0.0], [value]], dtype=dtype)
        normalizer.update(observations[:1])
        normalizer.update(observations[1:])
        expected = torch.tensor([value / 2], dtype=dtype)
        torch.testing.assert_close(normalizer.mean, expected)
        torch.testing.assert_close(normalizer.std, expected)
        torch.testing.assert_close(normalizer(observations), torch.tensor([[-1.0], [1.0]], dtype=dtype))

    def test_double_precision_unequal_batch_weights(self) -> None:
        """Integer sample counts must not round a double-precision running update through float32."""
        normalizer = EmpiricalNormalization(1).double()
        normalizer.update(torch.zeros(1, 1, dtype=torch.float64))
        normalizer.update(torch.ones(2, 1, dtype=torch.float64))
        torch.testing.assert_close(normalizer.mean, torch.tensor([2.0 / 3.0], dtype=torch.float64), rtol=1e-14, atol=0)
        torch.testing.assert_close(
            normalizer.std, torch.tensor([(2.0 / 9.0) ** 0.5], dtype=torch.float64), rtol=1e-14, atol=0
        )

    @pytest.mark.skipif(
        not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
        reason="Gloo is required for distributed normalization",
    )
    def test_distributed_weighted_batches_preserve_double_precision(self) -> None:
        """Two real Gloo ranks agree with the pooled moments through non-power-of-two update weights."""
        spawn_gloo(_double_moments_worker)

    def test_convergence_to_known_distribution(self) -> None:
        """Running mean and std should converge to the true values of the input distribution."""
        true_mean, true_std = 5.0, 2.0
        norm = EmpiricalNormalization(shape=4)
        norm.train()

        torch.manual_seed(0)
        for _ in range(200):
            batch = true_mean + true_std * torch.randn(64, 4)
            norm.update(batch)

        assert torch.allclose(norm.mean, torch.full((4,), true_mean), atol=0.15)
        assert torch.allclose(norm.std, torch.full((4,), true_std), atol=0.15)

    def test_forward_applies_normalization(self) -> None:
        """forward() should return (x - mean) / (std + eps), not the raw input."""
        norm = EmpiricalNormalization(shape=2, eps=0.01)
        norm.train()

        data = torch.tensor([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
        norm.update(data)

        result = norm(data)
        expected = (data - norm._mean) / (norm._std + norm.eps)
        assert torch.allclose(result, expected)

    def test_until_stops_updates(self) -> None:
        """After 'until' samples are seen, further updates must not change statistics."""
        norm = EmpiricalNormalization(shape=2, until=100)
        norm.train()

        for _ in range(10):
            norm.update(torch.randn(20, 2))

        assert norm.count >= 100
        mean_snapshot = norm._mean.clone()
        std_snapshot = norm._std.clone()

        for _ in range(10):
            norm.update(torch.randn(20, 2) + 100)

        assert torch.equal(norm._mean, mean_snapshot)
        assert torch.equal(norm._std, std_snapshot)

    def test_eval_mode_freezes_stats(self) -> None:
        """In eval mode, update() should be a no-op."""
        norm = EmpiricalNormalization(shape=3)
        norm.train()
        norm.update(torch.randn(50, 3))

        mean_before = norm._mean.clone()
        std_before = norm._std.clone()

        norm.eval()
        norm.update(torch.randn(50, 3) + 100)

        assert torch.equal(norm._mean, mean_before)
        assert torch.equal(norm._std, std_before)

    def test_inverse_round_trip(self) -> None:
        """inverse(forward(x)) should approximately recover x."""
        norm = EmpiricalNormalization(shape=4, eps=1e-2)
        norm.train()

        torch.manual_seed(42)
        for _ in range(50):
            norm.update(torch.randn(32, 4) * 3 + 5)

        x = torch.randn(16, 4) * 3 + 5
        recovered = norm.inverse(norm(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_single_sample_does_not_produce_nan(self) -> None:
        """Updating with a single sample should not produce NaN in mean or std."""
        norm = EmpiricalNormalization(shape=2)
        norm.train()
        norm.update(torch.tensor([[1.0, 2.0]]))

        assert not torch.any(torch.isnan(norm._mean))
        assert not torch.any(torch.isnan(norm._std))

    def test_distributed_update_combines_mean_and_variance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Distributed updates should match moments computed from samples on all ranks."""
        norm = EmpiricalNormalization(shape=1)
        local_data = torch.tensor([[0.0], [2.0]])
        remote_data = torch.tensor([[10.0], [14.0], [18.0]])
        expected = torch.cat((local_data, remote_data))
        call_count = 0

        def fake_all_reduce(value: torch.Tensor) -> None:
            nonlocal call_count
            if call_count == 0:
                value += remote_data.shape[0]
            elif call_count == 1:
                value += remote_data.sum(dim=0, keepdim=True)
            else:
                global_mean = expected.mean(dim=0, keepdim=True)
                remote_var = remote_data.var(dim=0, unbiased=False, keepdim=True)
                remote_mean = remote_data.mean(dim=0, keepdim=True)
                value += remote_data.shape[0] * (remote_var + (remote_mean - global_mean).square())
            call_count += 1

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

        norm.update(local_data)

        assert call_count == 3
        assert norm.count == expected.shape[0]
        assert torch.allclose(norm.mean, expected.mean(dim=0))
        assert torch.allclose(norm.std, expected.std(dim=0, unbiased=False))


class TestEmpiricalDiscountedVariationNormalization:
    """Tests for ``EmpiricalDiscountedVariationNormalization``."""

    def test_constant_rewards_produce_stable_normalization(self) -> None:
        """Constant rewards should converge to a stable normalization factor."""
        gamma = 0.99
        norm = EmpiricalDiscountedVariationNormalization(shape=[], gamma=gamma)
        norm.train()

        reward = torch.tensor([1.0])
        outputs = [norm(reward).item() for _ in range(200)]

        # After convergence, the output should be approximately constant
        last_10 = outputs[-10:]
        assert max(last_10) - min(last_10) < 0.1, "Normalization should stabilize for constant rewards"

    def test_fresh_eval_normalizer_preserves_reward(self) -> None:
        """An untrained normalizer in eval mode divides rewards by its unit std."""
        norm = EmpiricalDiscountedVariationNormalization(shape=[])
        norm.eval()
        reward = torch.tensor([5.0])
        torch.testing.assert_close(norm(reward), reward)
        assert norm.emp_norm.std == 1
        assert norm.emp_norm.count == 0

    def test_zero_std_returns_raw_reward(self) -> None:
        """When std is zero, forward should return the raw reward."""
        norm = EmpiricalDiscountedVariationNormalization(shape=[])
        norm.emp_norm.update(torch.tensor([5.0, 5.0]))
        norm.eval()
        reward = torch.tensor([5.0])
        result = norm(reward)
        assert norm.emp_norm.std == 0
        torch.testing.assert_close(result, reward)

    def test_in_place_scaling_of_output_does_not_change_history(self) -> None:
        """Scaling the returned raw reward in place should not modify the discounted average."""
        norm = EmpiricalDiscountedVariationNormalization(shape=[], gamma=0.5)
        norm.train()
        # Identical rewards give zero std, so the first call returns the raw reward
        result = norm(torch.tensor([2.0, 2.0]))
        result.mul_(3.0)
        assert torch.equal(norm.disc_avg.avg, torch.tensor([2.0, 2.0]))


def test_normalizer_standard_deviation_buffer_identity() -> None:
    """Normalizer updates preserve buffers referenced by captured CUDA graphs."""
    normalizer = EmpiricalNormalization(3)
    original = normalizer._std
    normalizer.update(torch.randn(8, 3))
    assert normalizer._std is original
