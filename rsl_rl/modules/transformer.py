import torch
from typing import Tuple


class Head(torch.nn.Module):
    """A single causal self-attention head."""

    def __init__(self, hidden_size: int, head_size: int):
        super().__init__()

        self.query = torch.nn.Linear(hidden_size, head_size)
        self.key = torch.nn.Linear(hidden_size, head_size)
        self.value = torch.nn.Linear(hidden_size, head_size)

    def forward(self, x: torch.Tensor):
        x = x.transpose(0, 1)
        _, S, F = x.shape  # (Batch, Sequence, Features)

        q = self.query(x)
        k = self.key(x)

        weight = q @ k.transpose(-1, -2) * F**-0.5  # shape: (B, S, S)
        weight.masked_fill(torch.tril(torch.ones(S, S, device=x.device)) == 0, float("-inf"))
        weight = torch.nn.functional.softmax(weight, dim=-1)

        v = self.value(x)  # shape: (B, S, F)
        out = (weight @ v).transpose(0, 1)  # shape: (S, B, F)

        return out


class MultiHead(torch.nn.Module):
    def __init__(self, hidden_size: int, head_count: int):
        super().__init__()

        assert hidden_size % head_count == 0, f"Multi-headed attention head size must be a multiple of the head count."

        self.heads = torch.nn.ModuleList([Head(hidden_size, hidden_size // head_count) for _ in range(head_count)])
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(x)

        return out


class Block(torch.nn.Module):
    def __init__(self, hidden_size: int, head_count: int):
        super().__init__()

        self.sa = MultiHead(hidden_size, head_count)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 4 * hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.ln2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sa(self.ln1(x)) + x
        out = self.ff(self.ln2(x)) + x

        return out


class Transformer(torch.nn.Module):
    """A Transformer-based recurrent module.

    The Transformer module is a recurrent module that uses a Transformer architecture to process the input sequence. It
    uses a hidden state to emulate RNN-like behavior.
    """

    def __init__(
        self, input_size, output_size, hidden_size, block_count: int = 6, context_length: int = 64, head_count: int = 8
    ):
        """
        Args:
            input_size (int): The size of the input.
            output_size (int): The size of the output.
            hidden_size (int): The size of the hidden layers.
            block_count (int): The number of Transformer blocks.
            context_length (int): The length of the context to consider when predicting the next token.
            head_count (int): The number of attention heads per block.
        """

        assert context_length % 2 == 0, f"Context length must be even."

        super().__init__()

        self.context_length = context_length
        self.hidden_size = hidden_size

        self.feature_proj = torch.nn.Linear(input_size, hidden_size)
        self.position_embedding = torch.nn.Embedding(context_length, hidden_size)
        self.blocks = torch.nn.Sequential(
            *[Block(hidden_size, head_count) for _ in range(block_count)],
            torch.nn.LayerNorm(hidden_size),
        )
        self.head = torch.nn.Linear(hidden_size, output_size)

    @property
    def num_layers(self):
        # Set num_layers to half the context length for simple torch.nn.LSTM compatibility. TODO: This is a bit hacky.
        return self.context_length // 2

    def step(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Computes Transformer output given the full context and the input.

        Args:
            x (torch.Tensor): The input tensor of shape (Sequence, Batch, Features).
            context (torch.Tensor): The context tensor of shape (Context Length, Batch, Features).
        Returns:
            A tuple of the output tensor of shape (Sequence, Batch, Features) and the updated context with the input
            features appended. The context has shape (Context Length, Batch, Features).
        """
        S = x.shape[0]

        # Project input to feature space and add to context.
        ft_x = self.feature_proj(x)
        context = torch.cat((context, ft_x), dim=0)[-self.context_length :]

        # Add positional embedding to context.
        ft_pos = self.position_embedding(torch.arange(self.context_length, device=x.device)).unsqueeze(1)
        x = context + ft_pos

        # Compute output from Transformer blocks.
        x = self.blocks(x)
        out = self.head(x)[-S:]

        return out, context

    def forward(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Computes Transformer output given the input and the hidden state which encapsulates the context."""
        if hidden_state is None:
            hidden_state = self.reset_hidden_state(x.shape[1], device=x.device)
        context = torch.cat(hidden_state, dim=0)

        out, context = self.step(x, context)

        hidden_state = context[: self.num_layers], context[self.num_layers :]

        return out, hidden_state

    def reset_hidden_state(self, batch_size: int, device="cpu"):
        hidden_state = torch.zeros((self.context_length, batch_size, self.hidden_size), device=device)
        hidden_state = hidden_state[: self.num_layers], hidden_state[self.num_layers :]

        return hidden_state
