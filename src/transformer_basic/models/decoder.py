import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .encoder import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Represents one layer of the Transformer decoder.
    It consists of a masked multi-head self-attention, a multi-head
    cross-attention over the encoder's output, and a position-wise
    feed-forward network.
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
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder layer.

        Args:
            tgt (torch.Tensor): Input to the decoder layer of shape (batch, tgt_seq_len, d_model).
            memory (torch.Tensor): The output from the encoder of shape (batch, src_seq_len, d_model).
            tgt_mask (torch.Tensor, optional): The mask for the target sequence (look-ahead mask).
            memory_mask (torch.Tensor, optional): The mask for the encoder output (padding mask).

        Returns:
            torch.Tensor: The output of the decoder layer of shape (batch, tgt_seq_len, d_model).
        """
        # 1. Masked Self-Attention sub-layer
        tgt_2, _ = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        # Add & Norm
        tgt = self.norm1(tgt + self.dropout1(tgt_2))

        # 2. Cross-Attention sub-layer
        # Query comes from the decoder, Key and Value from the encoder's memory
        tgt_2, _ = self.cross_attn(query=tgt, key=memory, value=memory, mask=memory_mask)
        # Add & Norm
        tgt = self.norm2(tgt + self.dropout2(tgt_2))

        # 3. Feed-Forward sub-layer
        tgt_2 = self.feed_forward(tgt)
        # Add & Norm
        tgt = self.norm3(tgt + self.dropout3(tgt_2))

        return tgt
