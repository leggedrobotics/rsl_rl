# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distribution modules."""

import math
import torch

from rsl_rl.modules.distribution import GaussianDistribution, HeteroscedasticGaussianDistribution


class TestGaussianDistribution:
    """Tests for ``GaussianDistribution``."""

    def test_log_prob_standard_normal(self) -> None:
        """log_prob at the mean of N(0,1) should equal -0.5*log(2*pi) per dimension, summed."""
        dim = 4
        dist = GaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")
        mean = torch.zeros(1, dim)
        dist.update(mean)

        log_p = dist.log_prob(torch.zeros(1, dim))
        expected = -0.5 * math.log(2 * math.pi) * dim
        assert torch.allclose(log_p, torch.tensor([expected]), atol=1e-5)

    def test_log_prob_nonzero_mean(self) -> None:
        """log_prob should decrease as the sample moves away from the mean."""
        dist = GaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar")
        mean = torch.tensor([[3.0, 3.0]])
        dist.update(mean)

        lp_at_mean = dist.log_prob(mean)
        lp_far = dist.log_prob(mean + 5.0)
        assert lp_at_mean > lp_far, "log_prob should be higher at the mean"

    def test_entropy_analytical(self) -> None:
        """Entropy should match the analytical formula 0.5 * sum(log(2*pi*e*std^2))."""
        dim = 3
        std_val = 2.0
        dist = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="scalar")
        dist.update(torch.zeros(1, dim))

        expected = 0.5 * dim * math.log(2 * math.pi * math.e * std_val**2)
        assert torch.allclose(dist.entropy, torch.tensor([expected]), atol=1e-4)

    def test_kl_divergence_analytical(self) -> None:
        """KL(N(0,1) || N(mu,sigma)) should match the closed-form KL for univariate Gaussians."""
        dist = GaussianDistribution(output_dim=1, init_std=1.0, std_type="scalar")
        mu_old, sigma_old = 0.0, 1.0
        mu_new, sigma_new = 1.0, 2.0

        old_params = (torch.tensor([[mu_old]]), torch.tensor([[sigma_old]]))
        new_params = (torch.tensor([[mu_new]]), torch.tensor([[sigma_new]]))

        kl = dist.kl_divergence(old_params, new_params)

        # Analytical KL: log(s2/s1) + (s1^2 + (m1-m2)^2) / (2*s2^2) - 0.5
        expected = math.log(sigma_new / sigma_old) + (sigma_old**2 + (mu_old - mu_new) ** 2) / (2 * sigma_new**2) - 0.5
        assert torch.allclose(kl, torch.tensor([expected]), atol=1e-5)

    def test_kl_divergence_identical_is_zero(self) -> None:
        """KL(p || p) should be zero."""
        dist = GaussianDistribution(output_dim=4, init_std=1.5, std_type="scalar")
        params = (torch.zeros(1, 4), torch.full((1, 4), 1.5))
        kl = dist.kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros(1), atol=1e-6)

    def test_scalar_vs_log_std_equivalence(self) -> None:
        """Scalar and log parameterizations should give identical results for the same effective std."""
        dim = 3
        std_val = 1.5
        dist_scalar = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="scalar")
        dist_log = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="log")

        mean = torch.randn(2, dim)
        dist_scalar.update(mean)
        dist_log.update(mean)

        sample_point = torch.randn(2, dim)

        assert torch.allclose(dist_scalar.log_prob(sample_point), dist_log.log_prob(sample_point), atol=1e-5)
        assert torch.allclose(dist_scalar.entropy, dist_log.entropy, atol=1e-5)

    def test_log_prob_gradient_flows_to_mean(self) -> None:
        """log_prob should allow gradient flow back to the distribution mean."""
        dim = 3
        dist = GaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")
        mean = torch.randn(1, dim, requires_grad=True)
        dist.update(mean)

        sample = dist.sample().detach()
        log_p = dist.log_prob(sample)
        log_p.sum().backward()
        assert mean.grad is not None, "Gradient should flow from log_prob to mean"
        assert not torch.all(mean.grad == 0), "Gradient should be non-zero"


class TestHeteroscedasticGaussianDistribution:
    """Tests for ``HeteroscedasticGaussianDistribution``."""

    def test_update_splits_mean_and_std(self) -> None:
        """update() should parse MLP output into separate mean and std."""
        dim = 4
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")

        mean_val = torch.randn(2, dim)
        std_val = torch.abs(torch.randn(2, dim)) + 0.1
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        dist.update(mlp_output)
        assert torch.allclose(dist.mean, mean_val, atol=1e-6)
        assert torch.allclose(dist.std, std_val, atol=1e-6)

    def test_deterministic_output_returns_mean(self) -> None:
        """deterministic_output() should extract the mean from the MLP output."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")

        mean_val = torch.tensor([[1.0, 2.0, 3.0]])
        std_val = torch.tensor([[0.5, 0.5, 0.5]])
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        result = dist.deterministic_output(mlp_output)
        assert torch.allclose(result, mean_val)

    def test_log_std_parameterization(self) -> None:
        """With std_type='log', the second slice should be treated as log(std)."""
        dim = 2
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="log")

        mean_val = torch.zeros(1, dim)
        log_std_val = torch.zeros(1, dim)  # log(1) = 0, so std = 1
        mlp_output = torch.stack([mean_val, log_std_val], dim=-2)

        dist.update(mlp_output)
        assert torch.allclose(dist.std, torch.ones(1, dim), atol=1e-6)

    def test_input_dim_is_pair(self) -> None:
        """input_dim should be [2, output_dim] to accommodate mean and std."""
        dim = 5
        dist = HeteroscedasticGaussianDistribution(output_dim=dim)
        assert dist.input_dim == [2, dim]
