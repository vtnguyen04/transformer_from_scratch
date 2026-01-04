import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism as described in
    "Attention Is All You Need".
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The dimensionality of the input and output.
            nhead (int): The number of attention heads.
            dropout (float): The dropout probability.
        """
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the scaled dot-product attention.

        Args:
            q (torch.Tensor): Queries tensor of shape (batch, nhead, seq_len_q, d_k).
            k (torch.Tensor): Keys tensor of shape (batch, nhead, seq_len_k, d_k).
            v (torch.Tensor): Values tensor of shape (batch, nhead, seq_len_v, d_k).
            mask (torch.Tensor, optional): Mask to be applied to the attention scores.
                                           Shape (batch, 1, 1, seq_len_k) or (1, 1, seq_len_q, seq_len_k).

        Returns:
            A tuple containing:
            - torch.Tensor: The context vector after attention.
            - torch.Tensor: The attention weights.
        """
        # (batch, nhead, seq_len_q, d_k) @ (batch, nhead, d_k, seq_len_k) -> (batch, nhead, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch, nhead, seq_len_q, seq_len_k) @ (batch, nhead, seq_len_v, d_k) -> (batch, nhead, seq_len_q, d_k)
        # Note: seq_len_k == seq_len_v
        context = torch.matmul(attn_weights, v)

        return context, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, seq_len_q, d_model).
            key (torch.Tensor): Key tensor of shape (batch, seq_len_k, d_model).
            value (torch.Tensor): Value tensor of shape (batch, seq_len_v, d_model).
            mask (torch.Tensor, optional): Mask to be applied. Its shape depends on whether it's a
                                           padding mask or a look-ahead mask.

        Returns:
            A tuple containing:
            - torch.Tensor: The output of the multi-head attention, shape (batch, seq_len_q, d_model).
            - torch.Tensor: The attention weights, shape (batch, nhead, seq_len_q, seq_len_k).
        """
        batch_size = query.size(0)

        # 1. Linear projections
        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)

        # 2. Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, nhead, d_k) -> (batch, nhead, seq_len, d_k)
        q = q.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 3. Apply scaled dot-product attention
        context, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 4. Concatenate heads and apply final linear layer
        # (batch, nhead, seq_len_q, d_k) -> (batch, seq_len_q, nhead, d_k)
        context = context.transpose(1, 2).contiguous()
        # (batch, seq_len_q, nhead * d_k) -> (batch, seq_len_q, d_model)
        context = context.view(batch_size, -1, self.d_model)

        output = self.w_o(context)

        return output, attn_weights
