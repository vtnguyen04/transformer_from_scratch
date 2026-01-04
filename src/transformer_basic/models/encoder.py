import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the
    embeddings, so that the two can be summed.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network (FFN) sub-layer.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model).
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Represents one layer of the Transformer encoder.
    It consists of a multi-head self-attention mechanism followed by a
    position-wise feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            src (torch.Tensor): Input to the encoder layer of shape (batch, seq_len, d_model).
            src_mask (torch.Tensor, optional): The mask for the source sequence.

        Returns:
            torch.Tensor: The output of the encoder layer of shape (batch, seq_len, d_model).
        """
        # 1. Self-Attention sub-layer
        # The query, key, and value are all the same: `src`
        src_2, _ = self.self_attn(src, src, src, mask=src_mask)
        # Add & Norm
        src = self.norm1(src + self.dropout1(src_2))

        # 2. Feed-Forward sub-layer
        src_2 = self.feed_forward(src)
        # Add & Norm
        src = self.norm2(src + self.dropout2(src_2))

        return src
