import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Callable, Tuple, Union

from rsl_rl.modules.network import Network
from rsl_rl.utils.benchmarkable import Benchmarkable
from rsl_rl.utils.utils import squeeze_preserve_batch

eps = torch.finfo(torch.float32).eps


def reshape_measure_parameters(
    qn: Network, *params: Union[torch.Tensor, float]
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Reshapes the parameters of a measure function to match the shape of the quantile network.

    Args:
        qn (Network): The quantile network.
        *params (Union[torch.Tensor, float]): The parameters of the measure function.
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]: The reshaped parameters.
    """
    if not params:
        return qn._tau.to(qn.device), *params

    assert len([*set([torch.is_tensor(p) for p in params])]) == 1, "All parameters must be either tensors or scalars."

    if torch.is_tensor(params[0]):
        assert all([p.dim() == 1 for p in params]), "All parameters must have dimensionality 1."
        assert len([*set([p.shape[0] for p in params])]) == 1, "All parameters must have the same size."

        reshaped_params = [p.reshape(-1, 1).to(qn.device) for p in params]
        tau = qn._tau.expand(params[0].shape[0], -1).to(qn.device)
    else:
        reshaped_params = params
        tau = qn._tau.to(qn.device)

    return tau, *reshaped_params


def make_distorted_measure(distorted_tau: torch.Tensor) -> Callable:
    """Creates a measure function for the distorted expectation under some distortion function.

    The distorted expectation for some distortion function g(tau) is given by the integral w.r.t. tau
    "int_0^1 g'(tau) * F_Z^{-1}(tau) dtau" where g'(tau) is the derivative of g w.r.t. tau and F_Z^{-1} is the inverse
    cumulative distribution function of the value distribution.
    See https://arxiv.org/pdf/2004.14547.pdf and https://arxiv.org/pdf/1806.06923.pdf for details.
    """
    distorted_tau = distorted_tau.reshape(-1, distorted_tau.shape[-1])
    distortion = (distorted_tau[:, 1:] - distorted_tau[:, :-1]).squeeze(0)

    def distorted_measure(quantiles):
        sorted_quantiles, _ = quantiles.sort(-1)
        sorted_quantiles = sorted_quantiles.reshape(-1, sorted_quantiles.shape[-1])

        # dtau = tau[1:] - tau[:-1] cancels the denominator of g'(tau) = g(tau)[1:] - g(tau)[:-1] / dtau.
        values = squeeze_preserve_batch((distortion.to(sorted_quantiles.device) * sorted_quantiles).sum(-1))

        return values

    return distorted_measure


def risk_measure_cvar(qn: Network, confidence_level: float = 1.0) -> Callable:
    """Conditional value at risk measure.

    TODO: Handle confidence_level being a tensor.

    Args:
        qn (QuantileNetwork): Quantile network to compute the risk measure for.
        confidence_level (float): Confidence level of the risk measure. Must be between 0 and 1.
    Returns:
        A risk measure function.
    """
    tau, confidence_level = reshape_measure_parameters(qn, confidence_level)
    distorted_tau = torch.min(tau / confidence_level, torch.ones(*tau.shape).to(tau.device))

    return make_distorted_measure(distorted_tau)


def risk_measure_neutral(_: Network) -> Callable:
    """Neutral risk measure (expected value).

    Args:
        _ (QuantileNetwork): Quantile network to compute the risk measure for.
    Returns:
        A risk measure function.
    """

    def measure(quantiles):
        values = squeeze_preserve_batch(quantiles.mean(-1))

        return values

    return measure


def risk_measure_percentile(_: Network, confidence_level: float = 1.0) -> Callable:
    """Value at risk measure.

    Args:
        _ (QuantileNetwork): Quantile network to compute the risk measure for.
        confidence_level (float): Confidence level of the risk measure. Must be between 0 and 1.
    Returns:
        A risk measure function.
    """

    def measure(quantiles):
        sorted_quantiles, _ = quantiles.sort(-1)
        sorted_quantiles = sorted_quantiles.reshape(-1, sorted_quantiles.shape[-1])
        idx = min(int(confidence_level * quantiles.shape[-1]), quantiles.shape[-1] - 1)

        values = squeeze_preserve_batch(sorted_quantiles[:, idx])

        return values

    return measure


def risk_measure_wang(qn: Network, beta: Union[float, torch.Tensor] = 0.0) -> Callable:
    """Wang's risk measure.

    The risk measure computes the distorted expectation under Wang's risk distortion function
    g(tau) = Phi(Phi^-1(tau) + beta) where Phi and Phi^-1 are the standard normal CDF and its inverse.
    See https://arxiv.org/pdf/2004.14547.pdf for details.

    Args:
        qn (QuantileNetwork): Quantile network to compute the risk measure for.
        beta (float): Parameter of the risk distortion function.
    Returns:
        A risk measure function.
    """
    tau, beta = reshape_measure_parameters(qn, beta)

    distorted_tau = Normal(0, 1).cdf(Normal(0, 1).icdf(tau) + beta)

    return make_distorted_measure(distorted_tau)


def energy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes sample energy loss between predictions and targets.

    The energy loss is computed as 2*E[||X - Y||_2] - E[||X - X'||_2] - E[||Y - Y'||_2], where X, X' and Y, Y' are
    random variables and ||.||_2 is the L2-norm. X, X' are the predictions and Y, Y' are the targets.

    Args:
        predictions (torch.Tensor): Predictions to compute loss from.
        targets (torch.Tensor): Targets to compare predictions against.
    Returns:
        A torch.Tensor of shape (1,) containing the loss.
    """
    dims = [-1 for _ in range(predictions.dim())]
    prediction_mat = predictions.unsqueeze(-1).expand(*dims, predictions.shape[-1])
    target_mat = targets.unsqueeze(-1).expand(*dims, predictions.shape[-1])

    delta_xx = (prediction_mat - prediction_mat.transpose(-1, -2)).abs().mean()
    delta_yy = (target_mat - target_mat.transpose(-1, -2)).abs().mean()
    delta_xy = (prediction_mat - target_mat.transpose(-1, -2)).abs().mean()

    loss = 2 * delta_xy - delta_xx - delta_yy

    return loss


class QuantileNetwork(Network):
    measure_cvar = "cvar"
    measure_neutral = "neutral"
    measure_percentile = "percentile"
    measure_wang = "wang"

    measures = {
        measure_cvar: risk_measure_cvar,
        measure_neutral: risk_measure_neutral,
        measure_percentile: risk_measure_percentile,
        measure_wang: risk_measure_wang,
    }

    def __init__(
        self,
        input_size,
        output_size,
        activations=["relu", "relu", "relu"],
        hidden_dims=[256, 256, 256],
        init_fade=False,
        init_gain=0.5,
        measure=None,
        measure_kwargs={},
        quantile_count=200,
        **kwargs,
    ):
        assert len(hidden_dims) == len(activations)
        assert quantile_count > 0

        super().__init__(
            input_size,
            activations=activations,
            hidden_dims=hidden_dims[:-1],
            init_fade=False,
            init_gain=init_gain,
            output_size=hidden_dims[-1],
            **kwargs,
        )

        self._quantile_count = quantile_count
        self._tau = torch.arange(self._quantile_count + 1) / self._quantile_count
        self._tau_hat = torch.tensor([(self._tau[i] + self._tau[i + 1]) / 2 for i in range(self._quantile_count)])
        self._tau_hat_mat = torch.empty((0,))

        self._quantile_layers = nn.ModuleList([nn.Linear(hidden_dims[-1], quantile_count) for _ in range(output_size)])

        self._init(self._quantile_layers, fade=init_fade, gain=init_gain)

        measure_func = risk_measure_neutral if measure is None else self.measures[measure]
        self._measure_func = measure_func
        self._measure = measure_func(self, **measure_kwargs)

        self._last_quantiles = None

    @property
    def last_quantiles(self) -> torch.Tensor:
        return self._last_quantiles

    def make_diracs(self, values: torch.Tensor) -> torch.Tensor:
        """Generates value distributions that have a single spike at the given values.

        Args:
            values (torch.Tensor): Values to generate dirac distributions for.
        Returns:
            A torch.Tensor of shape (*values.shape, quantile_count) containing the dirac distributions.
        """
        dirac = values.unsqueeze(-1).expand(*[-1 for _ in range(values.dim())], self._quantile_count)

        return dirac

    @property
    def quantile_count(self) -> int:
        return self._quantile_count

    @Benchmarkable.register
    def quantiles_to_values(self, quantiles: torch.Tensor, *measure_args) -> torch.Tensor:
        """Computes values from quantiles.

        Args:
            quantiles (torch.Tensor): Quantiles to compute values from.
            measure_kwargs (dict): Keyword arguments to pass to the risk measure function instead of the arguments
                passed when creating the network.
        Returns:
            A torch.Tensor of shape (1,) containing the values.
        """
        if measure_args:
            values = self._measure_func(self, *[squeeze_preserve_batch(m) for m in measure_args])(quantiles)
        else:
            values = self._measure(quantiles)

        return values

    @Benchmarkable.register
    def quantile_l1_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes quantile-wise l1 loss between predictions and targets.

        TODO: This function is a bottleneck.

        Args:
            predictions (torch.Tensor): Predictions to compute loss from.
            targets (torch.Tensor): Targets to compare predictions against.
        Returns:
            A torch.Tensor of shape (1,) containing the loss.
        """
        assert (
            predictions.dim() == 2 or predictions.dim() == 3
        ), f"Predictions must be 2D or 3D. Got {predictions.dim()}."
        assert (
            predictions.shape == targets.shape
        ), f"The shapes of predictions and targets must match. Got {predictions.shape} and {targets.shape}."

        pre_dims = [-1] if predictions.dim() == 3 else []

        prediction_mat = predictions.unsqueeze(-3).expand(*pre_dims, self._quantile_count, -1, -1)
        target_mat = targets.transpose(-2, -1).unsqueeze(-1).expand(*pre_dims, -1, -1, self._quantile_count)
        delta = target_mat - prediction_mat

        tau_hat = self._tau_hat.expand(predictions.shape[-2], -1).to(self.device)
        loss = (torch.where(delta < 0, (tau_hat - 1), tau_hat) * delta).abs().mean()

        return loss

    @Benchmarkable.register
    def quantile_huber_loss(self, predictions: torch.Tensor, targets: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """Computes quantile huber loss between predictions and targets.

        TODO: This function is a bottleneck.

        Args:
            predictions (torch.Tensor): Predictions to compute loss from.
            targets (torch.Tensor): Targets to compare predictions against.
            kappa (float): Defines the interval [-kappa, kappa] around zero where squared loss is used. Defaults to 1.
        Returns:
            A torch.tensor of shape (1,) containing the loss.
        """
        pre_dims = [-1] if predictions.dim() == 3 else []

        prediction_mat = predictions.unsqueeze(-3).expand(*pre_dims, self._quantile_count, -1, -1)
        target_mat = targets.transpose(-2, -1).unsqueeze(-1).expand(*pre_dims, -1, -1, self._quantile_count)
        delta = target_mat - prediction_mat
        delta_abs = delta.abs()

        huber = torch.where(delta_abs <= kappa, 0.5 * delta.pow(2), kappa * (delta_abs - 0.5 * kappa))

        tau_hat = self._tau_hat.expand(predictions.shape[-2], -1).to(self.device)
        loss = (torch.where(delta < 0, (tau_hat - 1), tau_hat).abs() * huber).mean()

        return loss

    @Benchmarkable.register
    def sample_energy_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = energy_loss(predictions, targets)

        return loss

    @Benchmarkable.register
    def forward(self, x: torch.Tensor, distribution: bool = False, measure_args: list = [], **kwargs) -> torch.Tensor:
        features = super().forward(x, **kwargs)
        quantiles = squeeze_preserve_batch(torch.stack([layer(features) for layer in self._quantile_layers], dim=1))

        self._last_quantiles = quantiles

        if distribution:
            return quantiles

        values = self.quantiles_to_values(quantiles, *measure_args)

        return values
