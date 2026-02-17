# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distribution modules for stochastic models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class Distribution(nn.Module):
    """Base class for distribution modules.

    Distribution modules encapsulate the stochastic output of a neural model. They define the output structure expected
    from the MLP, manage learnable distribution parameters, and provide methods for sampling, log probability
    computation, and entropy calculation.

    Subclasses must implement all abstract methods and properties to define a specific distribution type.
    """

    def __init__(self, output_dim: int) -> None:
        """Initialize the distribution module.

        Args:
            output_dim: Dimension of the action/output space.
        """
        super().__init__()
        self.output_dim = output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution parameters given the MLP output.

        Args:
            mlp_output: Raw output from the MLP.
        """
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """Sample from the distribution.

        Returns:
            Sampled values.
        """
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the deterministic (mean) output from the raw MLP output.

        Args:
            mlp_output: Raw output from the MLP.

        Returns:
            The deterministic output (typically the distribution mean).
        """
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        """Return the input dimension required by the distribution."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation (or spread measure) of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the distribution, summed over the last dimension."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the distribution parameters as a tuple of tensors.

        These are the distribution-specific parameters needed to reconstruct the distribution (e.g., mean and std for
        Gaussian, alpha and beta for Beta). They are stored during rollouts and used for KL divergence computation.
        """
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given outputs, summed over the last dimension.

        Args:
            outputs: Values to compute the log probability for.

        Returns:
            Log probability summed over the last dimension.
        """
        raise NotImplementedError

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the KL divergence KL(old || new) between two distributions of this type.

        The KL divergence measures how the old distribution diverges from the new distribution.
        This is used for adaptive learning rate scheduling in policy optimization.

        Args:
            old_params: Parameters of the old distribution (as returned by :attr:`params`).
            new_params: Parameters of the new distribution (as returned by :attr:`params`).

        Returns:
            KL divergence summed over the last dimension.
        """
        raise NotImplementedError

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize distribution-specific weights in the MLP.

        This is called after MLP creation to set up any special weight initialization
        required by the distribution (e.g., initializing std head weights).

        Args:
            mlp: The MLP module whose weights may need initialization.
        """
        pass


class GaussianDistribution(Distribution):
    """Gaussian (Normal) distribution module with state-independent standard deviation.

    This distribution parameterizes actions using a multivariate Gaussian with diagonal covariance. The standard
    deviation is a learnable parameter that is independent of the model input. It can be parameterized in either
    "scalar" space (directly) or "log" space.
    """

    def __init__(
        self,
        output_dim: int,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_noise_std: Initial standard deviation.
            noise_std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        super().__init__(output_dim)
        self.noise_std_type = noise_std_type

        # Learnable std parameters
        if noise_std_type == "scalar":
            self.std_param = nn.Parameter(init_noise_std * torch.ones(output_dim))
        elif noise_std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'.")

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        mean = mlp_output
        if self.noise_std_type == "scalar":
            std = self.std_param.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output."""
        return mlp_output

    @property
    def input_dim(self) -> int:
        """Return the input dimension required by the distribution."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Gaussian distribution."""
        return self._distribution.mean  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Gaussian distribution."""
        return self._distribution.stddev  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the Gaussian distribution, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) of the current Gaussian distribution."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability under the Gaussian, summed over the last dimension."""
        return self._distribution.log_prob(outputs).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between two Gaussian distributions using torch.distributions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class HeteroscedasticGaussianDistribution(GaussianDistribution):
    """Gaussian (Normal) distribution module with state-dependent standard deviation.

    This distribution parameterizes actions using a multivariate Gaussian with diagonal covariance. The standard
    deviation is output by the MLP alongside the mean, making it state-dependent (heteroscedastic). It can be
    parameterized in either "scalar" space (directly) or "log" space.
    """

    def __init__(
        self,
        output_dim: int,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
    ) -> None:
        """Initialize the heteroscedastic Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_noise_std: Initial standard deviation (used to initialize MLP std head bias).
            noise_std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        # Skip GaussianDistribution.__init__ to avoid creating unnecessary learnable std parameters.
        Distribution.__init__(self, output_dim)
        self.noise_std_type = noise_std_type
        self.init_noise_std = init_noise_std

        if noise_std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'.")

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        if self.noise_std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
        elif self.noise_std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output (first slice of the second-to-last dimension)."""
        return mlp_output[..., 0, :]

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution.

        The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
        dimension is the mean and the second is the standard deviation (or log standard deviation).
        """
        return [2, self.output_dim]

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the std head weights in the MLP."""
        # Initialize weights and biases for the std portion of the last layer
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        if self.noise_std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_noise_std)  # type: ignore
        elif self.noise_std_type == "log":
            init_noise_std_log = torch.log(torch.tensor(self.init_noise_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], init_noise_std_log)  # type: ignore
