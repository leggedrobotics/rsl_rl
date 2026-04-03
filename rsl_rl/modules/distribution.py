# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the deterministic output from the MLP output."""
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
        init_std: float = 1.0,
        std_type: str = "scalar",
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        super().__init__(output_dim)
        self.std_type = std_type

        # Learnable std parameters
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.expand_as(mean)
        elif self.std_type == "log":
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the mean from the MLP output."""
        return _IdentityDeterministicOutput()

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
        init_std: float = 1.0,
        std_type: str = "scalar",
    ) -> None:
        """Initialize the heteroscedastic Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation (used to initialize MLP std head bias).
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        # Skip GaussianDistribution.__init__ to avoid creating unnecessary learnable std parameters.
        Distribution.__init__(self, output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output (first slice of the second-to-last dimension)."""
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that extracts the mean from the MLP output."""
        return _MeanSliceDeterministicOutput()

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
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_std)  # type: ignore
        elif self.std_type == "log":
            init_std_log = torch.log(torch.tensor(self.init_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], init_std_log)  # type: ignore


class SquashedGaussianDistribution(Distribution):
    """Tanh-squashed Gaussian distribution for bounded continuous actions."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "log",
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__(output_dim)
        self.std_type = std_type
        self.init_std = init_std
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        self._distribution: Normal | None = None
        self._mean: torch.Tensor | None = None
        self._log_std: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
            std = torch.clamp(std, min=1e-6)
            log_std = torch.log(std)
        else:
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
            std = torch.exp(log_std)

        self._mean = mean
        self._log_std = log_std
        self._std = std
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        pre_tanh = self._distribution.sample()  # type: ignore
        return torch.tanh(pre_tanh)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        mean = mlp_output[..., 0, :]
        return torch.tanh(mean)

    def as_deterministic_output_module(self) -> nn.Module:
        return _TanhMeanDeterministicOutput()

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self._mean)  # type: ignore[arg-type]

    @property
    def std(self) -> torch.Tensor:
        return self._std  # type: ignore[return-value]

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)  # type: ignore[union-attr]

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._mean, self._log_std)  # type: ignore[return-value]

    def sample_with_log_prob(self, reparameterize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if reparameterize:
            pre_tanh = self._distribution.rsample()  # type: ignore[union-attr]
        else:
            pre_tanh = self._distribution.sample()  # type: ignore[union-attr]
        actions = torch.tanh(pre_tanh)
        log_prob = self._distribution.log_prob(pre_tanh)  # type: ignore[union-attr]
        log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6)
        return actions, log_prob.sum(dim=-1, keepdim=True)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.clamp(outputs, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = self._atanh(outputs)
        log_prob = self._distribution.log_prob(pre_tanh)  # type: ignore[union-attr]
        log_prob -= torch.log(1.0 - outputs.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        old_mean, old_log_std = old_params
        new_mean, new_log_std = new_params
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_std)  # type: ignore
        elif self.std_type == "log":
            init_std_log = torch.log(torch.tensor(self.init_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], init_std_log)  # type: ignore

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class _IdentityDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output as is."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _MeanSliceDeterministicOutput(nn.Module):
    """Exportable module that extracts the mean from the MLP output (first slice of the second-to-last dimension)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]


class _TanhMeanDeterministicOutput(nn.Module):
    """Exportable module that applies tanh to the mean slice of MLP outputs."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output[..., 0, :])
