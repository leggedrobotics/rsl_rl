import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-6, until=None) -> None:
        """
        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
                exceeds it.
        """
        super().__init__()

        self.eps = eps
        self.until = until

        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))

        self.count = 0

    @property
    def mean(self) -> torch.Tensor:
        """Mean of input values."""
        return self._mean.squeeze(0).detach().clone()

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation of input values."""
        return self._std.squeeze(0).detach().clone()

    def forward(self, x) -> torch.Tensor:
        """Normalize mean and variance of values based on emprical values.
        Args:
            x (ndarray or Variable): Input values
        Returns:
            Normalized output values
        """

        if self.training:
            self.update(x)

        x_normalized = (x - self._mean.detach()) / (self._std.detach() + self.eps)

        return x_normalized

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them.

        Args:
            x (torch.Tensor): Input values.
        """
        x = x.detach()

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse normalized values.

        Args:
            y (torch.Tensor): Normalized input values.
        Returns:
            Inverse normalized output values.
        """
        inv = y * (self._std + self.eps) + self._mean

        return inv
