import torch
import torch.nn as nn
import math
import copy
from .encoder import EncoderLayer, PositionalEncoding
from .decoder import DecoderLayer

class Transformer(nn.Module):
    """
    An implementation of the Transformer model from "Attention Is All You Need".
    This implementation uses manually stacked encoder and decoder layers.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        Assumes batch_first = True.
        """
        # Reshape masks for multi-head attention
        # src_padding_mask: [batch, src_len] -> [batch, 1, 1, src_len]
        # This is for broadcasting over heads and query positions.
        src_mask_att = src_padding_mask.unsqueeze(1).unsqueeze(2)

        # tgt_mask (look-ahead) is [tgt_len, tgt_len]
        # tgt_padding_mask is [batch, tgt_len]
        # We combine them for the decoder's self-attention.
        tgt_padding_mask_att = tgt_padding_mask.unsqueeze(1).unsqueeze(2)
        # combined_tgt_mask shape will be broadcastable to [batch, nhead, tgt_len, tgt_len]
        combined_tgt_mask = tgt_mask.unsqueeze(0) | tgt_padding_mask_att

        memory = self.encode(src, src_mask_att)
        output = self.decode(tgt, memory, combined_tgt_mask, src_mask_att)
        return self.generator(output)

    def encode(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encodes the source sequence.
        Input shape: (batch, seq_len)
        """
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        output = self.pos_encoder(src_emb)
        
        for layer in self.encoder_layers:
            output = layer(output, src_mask=mask)
        
        return output

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Decodes the target sequence based on encoder's memory.
        Input shape: (batch, seq_len)
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        output = self.pos_encoder(tgt_emb)
        
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            
        return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates a square causal mask for the sequence.
    The masked positions are True.
    """
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

