# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories

HiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
"""Type alias for the hidden state of RNNs (GRU/LSTM).

For GRUs, this is a single tensor while for LSTMs, this is a tuple of two tensors (hidden state and cell state).
"""


class Memory(nn.Module):
    """Memory network for recurrent architectures.

    This network is used to store the hidden state of the policy. It currently supports GRU and LSTM.
    """

    def __init__(self, input_size: int, hidden_dim: int = 256, num_layers: int = 1, type: str = "lstm") -> None:
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden_state = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:
            # Batch mode needs saved hidden states
            if hidden_state is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_state)
            out = unpad_trajectories(out, masks)
        else:
            # Inference/distillation mode uses hidden state of last step
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:  # Reset hidden state
            if hidden_state is None:
                self.hidden_state = None
            else:
                self.hidden_state = hidden_state
        elif self.hidden_state is not None:  # Reset hidden state of done environments
            if hidden_state is None:
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_state[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting the hidden state of done environments with a custom hidden state is not implemented"
                )

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is not None:
            if dones is None:  # Detach hidden state
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    self.hidden_state = tuple(hidden_state.detach() for hidden_state in self.hidden_state)
                else:
                    self.hidden_state = self.hidden_state.detach()
            else:  # Detach hidden state of done environments
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()
