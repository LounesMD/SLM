import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.utils import (
    FFNetwork,
    MaskedMulltiHeadAttention,
    MulltiHeadAttention,
    pos_embeddings,
)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super(TransformerDecoder, self).__init__()
        self.device = torch.device("mps")
        self.norm1 = nn.LayerNorm(n_embed)
        self.multi_head = MulltiHeadAttention(
            n_heads, n_embed, head_size, block_size, dropout
        )
        self.masked_multi_head = MaskedMulltiHeadAttention(
            n_heads, n_embed, head_size, block_size, dropout
        )
        self.norm2 = nn.LayerNorm(n_embed)
        self.ff_net = FFNetwork(n_embed, dropout)
        self.norm3 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Pre-norm
        if isinstance(x, tuple):
            x, enc_out = x
        else:
            x, enc_out = x, None

        x = x + self.masked_multi_head(self.norm1(x))
        x = x + self.multi_head(self.norm2(x), enc_out=enc_out)
        x = x + self.ff_net(self.norm3(x))
        return x
