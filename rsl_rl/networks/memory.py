# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories


class Memory(nn.Module):
    """Memory module for recurrent networks.

    This module is used to store the hidden states of the policy. It currently only supports GRU and LSTM.
    """

    def __init__(self, input_size: int, hidden_dim: int = 256, num_layers: int = 1, type: str = "lstm") -> None:
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.hidden_states = None

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | tuple[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:
            # Batch mode needs saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # Inference/distillation mode uses hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: torch.Tensor | tuple[torch.Tensor] | None = None
    ) -> None:
        if dones is None:  # Reset hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # Reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_states is not None:
            if dones is None:  # Detach all hidden states
                if isinstance(self.hidden_states, tuple):  # Tuple in case of LSTM
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:  # Detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()
