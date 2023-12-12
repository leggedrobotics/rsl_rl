import torch
import torch.nn as nn
from typing import List

from rsl_rl.modules.normalizer import EmpiricalNormalization
from rsl_rl.modules.utils import get_activation
from rsl_rl.modules.transformer import Transformer
from rsl_rl.utils.benchmarkable import Benchmarkable
from rsl_rl.utils.utils import squeeze_preserve_batch


class Network(Benchmarkable, nn.Module):
    recurrent_module_lstm = "LSTM"
    recurrent_module_transformer = "TF"

    recurrent_modules = {recurrent_module_lstm: nn.LSTM, recurrent_module_transformer: Transformer}

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activations: List[str] = ["relu", "relu", "relu", "tanh"],
        hidden_dims: List[int] = [256, 256, 256],
        init_fade: bool = True,
        init_gain: float = 1.0,
        input_normalization: bool = False,
        recurrent: bool = False,
        recurrent_layers: int = 1,
        recurrent_module: str = recurrent_module_lstm,
        recurrent_tf_context_length: int = 64,
        recurrent_tf_head_count: int = 8,
    ) -> None:
        """

        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
            activations (List[str]): The activation functions to use. If the network is recurrent, the first activation
                function is used for the output of the recurrent layer.
            hidden_dims (List[int]): The hidden dimensions. If the network is recurrent, the first hidden dimension is
                used for the recurrent layer.
            init_fade (bool): Whether to use the fade in initialization.
            init_gain (float): The gain to use for the initialization.
            input_normalization (bool): Whether to use input normalization.
            recurrent (bool): Whether to use a recurrent network.
            recurrent_layers (int): The number of recurrent layers (LSTM) / blocks (Transformer) to use.
            recurrent_module (str): The recurrent module to use. Must be one of Network.recurrent_modules.
            recurrent_tf_context_length (int): The context length of the Transformer.
            recurrent_tf_head_count (int): The head count of the Transformer.
        """
        assert len(hidden_dims) + 1 == len(activations)

        super().__init__()

        if input_normalization:
            self._normalization = EmpiricalNormalization(shape=(input_size,))
        else:
            self._normalization = nn.Identity()

        dims = [input_size] + hidden_dims + [output_size]

        self._recurrent = recurrent
        self._recurrent_module = recurrent_module
        self.hidden_state = None
        self._last_hidden_state = None
        if self._recurrent:
            recurrent_kwargs = dict()

            if recurrent_module == self.recurrent_module_lstm:
                recurrent_kwargs["hidden_size"] = dims[1]
                recurrent_kwargs["input_size"] = dims[0]
                recurrent_kwargs["num_layers"] = recurrent_layers
            elif recurrent_module == self.recurrent_module_transformer:
                recurrent_kwargs["block_count"] = recurrent_layers
                recurrent_kwargs["context_length"] = recurrent_tf_context_length
                recurrent_kwargs["head_count"] = recurrent_tf_head_count
                recurrent_kwargs["hidden_size"] = dims[1]
                recurrent_kwargs["input_size"] = dims[0]
                recurrent_kwargs["output_size"] = dims[1]

            rnn = self.recurrent_modules[recurrent_module](**recurrent_kwargs)
            activation = get_activation(activations[0])
            dims = dims[1:]
            activations = activations[1:]

            self._features = nn.Sequential(rnn, activation)
        else:
            self._features = nn.Identity()

        layers = []
        for i in range(len(activations)):
            layer = nn.Linear(dims[i], dims[i + 1])
            activation = get_activation(activations[i])

            layers.append(layer)
            layers.append(activation)

        self._layers = nn.Sequential(*layers)

        if len(layers) > 0:
            self._init(self._layers, fade=init_fade, gain=init_gain)

    @property
    def device(self):
        """Returns the device of the network."""
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor, hidden_state=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input data.
            hidden_state (Tuple[torch.Tensor, torch.Tensor]): The hidden state of the network. If None, the hidden state
                of the network is used. If provided, the hidden state of the neural network will not be updated. To
                retrieve the new hidden state, use the last_hidden_state property. If the network is not recurrent,
                this argument is ignored.
        Returns:
            The output of the network as a torch.Tensor.
        """
        assert hidden_state is None or self._recurrent, "Cannot pass hidden state to non-recurrent network."

        input = self._normalization(x.to(self.device))

        if self._recurrent:
            current_hidden_state = self.hidden_state if hidden_state is None else hidden_state
            current_hidden_state = (current_hidden_state[0].to(self.device), current_hidden_state[1].to(self.device))

            input = input.unsqueeze(0) if len(input.shape) == 2 else input
            input, next_hidden_state = self._features[0](input, current_hidden_state)
            input = self._features[1](input).squeeze(0)

            if hidden_state is None:
                self.hidden_state = next_hidden_state
            self._last_hidden_state = next_hidden_state

        output = squeeze_preserve_batch(self._layers(input))

        return output

    @property
    def last_hidden_state(self):
        """Returns the hidden state of the last forward pass.

        Does not differentiate whether the hidden state depends on the hidden state kept in the network or whether it
        was passed into the forward pass.

        Returns:
            The hidden state of the last forward pass as Tuple[torch.Tensor, torch.Tensor].
        """
        return self._last_hidden_state

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the given input.

        Args:
            x (torch.Tensor): The input to normalize.
        Returns:
            The normalized input as a torch.Tensor.
        """
        output = self._normalization(x.to(self.device))

        return output

    @property
    def recurrent(self) -> bool:
        """Returns whether the network is recurrent."""
        return self._recurrent

    def reset_hidden_state(self, indices: torch.Tensor) -> None:
        """Resets the hidden state of the neural network.

        Throws an error if the network is not recurrent.

        Args:
            indices (torch.Tensor): A 1-dimensional int tensor containing the indices of the terminated
                environments.
        """
        assert self._recurrent

        self.hidden_state[0][:, indices] = torch.zeros(len(indices), self._features[0].hidden_size, device=self.device)
        self.hidden_state[1][:, indices] = torch.zeros(len(indices), self._features[0].hidden_size, device=self.device)

    def reset_full_hidden_state(self, batch_size=None) -> None:
        """Resets the hidden state of the neural network.

        Args:
            batch_size (int): The batch size of the hidden state. If None, the hidden state is reset to None.
        """
        assert self._recurrent

        if batch_size is None:
            self.hidden_state = None
        else:
            layer_count, hidden_size = self._features[0].num_layers, self._features[0].hidden_size
            self.hidden_state = (
                torch.zeros(layer_count, batch_size, hidden_size, device=self.device),
                torch.zeros(layer_count, batch_size, hidden_size, device=self.device),
            )

    def _init(self, layers: List[nn.Module], fade: bool = True, gain: float = 1.0) -> List[nn.Module]:
        """Initializes neural network layers."""
        last_layer_idx = len(layers) - 1 - next(i for i, l in enumerate(reversed(layers)) if isinstance(l, nn.Linear))

        for idx, layer in enumerate(layers):
            if not isinstance(layer, nn.Linear):
                continue

            current_gain = gain / 100.0 if fade and idx == last_layer_idx else gain
            nn.init.xavier_normal_(layer.weight, gain=current_gain)

        return layers
