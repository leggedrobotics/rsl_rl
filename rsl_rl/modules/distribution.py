# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal


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
    """Gaussian distribution module with state-independent standard deviation.

    This distribution parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation can be a learnable parameter or a constant. It can be parameterized in either "scalar" space or
    "log" space and is clamped to a specified range.

    .. note::
        If the standard deviation type is set to "log", the provided arguments are still interpreted in scalar space,
        and converted to log space internally.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_range: Range for the standard deviation. Should be a tuple of (min, max) values for clamping.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
            learn_std: Whether the standard deviation should be learnable. If False, it will be fixed to `init_std`.
        """
        super().__init__(output_dim)
        self.std_type = std_type

        # Learnable std parameters
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)), requires_grad=learn_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Clamp the std range to ensure numerical stability and store log space range if needed
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)  # Avoid zero std for numerical stability
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.clamp(self.std_range[0], self.std_range[1])
        elif self.std_type == "log":
            log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
            std = torch.exp(log_std)
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
        """Compute KL(old || new) between two Gaussian distributions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(dim=-1)


class HeteroscedasticGaussianDistribution(GaussianDistribution):
    """Gaussian distribution module with state-dependent standard deviation.

    This distribution parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation is output by the MLP alongside the mean, making it state-dependent. It can be parameterized in
    either "scalar" space or "log" space, and is clamped to a specified range.

    .. note::
        If the standard deviation type is set to "log", the provided arguments are still interpreted in scalar space,
        and converted to log space internally.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
    ) -> None:
        """Initialize the heteroscedastic Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation (used to initialize the MLP's std head bias).
            std_range: Range for the standard deviation. Should be a tuple of (min, max) values for clamping.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        # Skip GaussianDistribution.__init__ to avoid creating unnecessary learnable std parameters.
        Distribution.__init__(self, output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Clamp the std range to ensure numerical stability and store log space range if needed
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)  # Avoid zero std for numerical stability
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
            std = torch.clamp(std, self.std_range[0], self.std_range[1])
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            log_std = torch.clamp(log_std, self.log_std_range[0], self.log_std_range[1])
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output (first slice of the second-to-last dimension)."""
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that extracts the mean from the MLP output."""
        return _FirstSliceDeterministicOutput()

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


class BetaDistribution(Distribution):
    """Beta distribution module for bounded action spaces.

    This distribution parameterizes stochastic outputs using a Beta distribution, which naturally constrains samples
    to [0, 1]. Samples are linearly rescaled to ``action_range``, which defaults to ``(-1.0, 1.0)``.

    The MLP must output a tensor of shape ``[..., 2, output_dim]``, where the first slice along the second-to-last
    dimension contains the raw alpha parameters and the second contains the raw beta parameters. Both are passed
    through ``Softplus + 1`` to ensure they are strictly greater than 1, which guarantees a unimodal distribution.
    """

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Initialize the Beta distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            action_range: Interval ``(min, max)`` to which Beta samples in ``[0, 1]`` are linearly rescaled.
                Defaults to ``(-1.0, 1.0)``.
        """
        super().__init__(output_dim)

        # Compute scaling and offset for rescaling samples
        self.action_range = action_range
        self._range_scale = action_range[1] - action_range[0]
        self._range_offset = action_range[0]
        self._log_range_scale = np.log(self._range_scale)

        self._distribution: Beta | None = None
        self._alpha: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None

        # Disable args validation for speedup
        Beta.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Beta distribution from MLP output."""
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        self._alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        self._beta = torch.nn.functional.softplus(beta_raw) + 1.0
        self._distribution = Beta(self._alpha, self._beta)

    def sample(self) -> torch.Tensor:
        """Sample from the Beta distribution and rescale to ``action_range``."""
        return self._distribution.sample() * self._range_scale + self._range_offset  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output and rescale to ``action_range``."""
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return (alpha / (alpha + beta)) * self._range_scale + self._range_offset

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that computes the mean from the MLP output."""
        return _BetaDeterministicOutput(self._range_scale, self._range_offset)

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution.

        The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
        dimension is the raw alpha parameter and the second is the raw beta parameter.
        """
        return [2, self.output_dim]

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Beta distribution, rescaled to ``action_range``."""
        return (self._alpha / (self._alpha + self._beta)) * self._range_scale + self._range_offset  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Beta distribution, rescaled to ``action_range``."""
        return self._distribution.stddev * self._range_scale  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the Beta distribution, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return ``(alpha, beta)`` of the current Beta distribution."""
        return (self._alpha, self._beta)  # type: ignore

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability under the Beta distribution, summed over the last dimension.

        Outputs are unscaled from ``action_range`` back to ``[0, 1]`` before computing the log probability.
        The Jacobian correction for the linear rescaling is included.
        """
        unscaled = (outputs - self._range_offset) / self._range_scale
        unscaled = unscaled.clamp(1e-6, 1.0 - 1e-6)
        # Jacobian correction: log p(y) = log p(x) - log(scale), where y = x * scale + offset
        return (self._distribution.log_prob(unscaled) - self._log_range_scale).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between two Beta distributions."""
        old_alpha, old_beta = old_params
        new_alpha, new_beta = new_params
        return torch.distributions.kl_divergence(Beta(old_alpha, old_beta), Beta(new_alpha, new_beta)).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the beta-parameter head weights to zero for a near-uniform initial distribution."""
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        torch.nn.init.zeros_(mlp[-2].bias[self.output_dim :])  # type: ignore


class _IdentityDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output as is."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _FirstSliceDeterministicOutput(nn.Module):
    """Exportable module that extracts the mean from the MLP output (first slice of the second-to-last dimension)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]


class _BetaDeterministicOutput(nn.Module):
    """Exportable module that computes the mean of the Beta distribution from the MLP output."""

    def __init__(self, range_scale: float, range_offset: float) -> None:
        super().__init__()
        self.range_scale = range_scale
        self.range_offset = range_offset

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        alpha_raw, beta_raw = torch.unbind(mlp_output, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return (alpha / (alpha + beta)) * self.range_scale + self.range_offset
