import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.utils import FFNetwork, MulltiHeadAttention, pos_embeddings


class TransformerEncoder(nn.Module):
    """
    Transformer encoder
    """

    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super(TransformerEncoder, self).__init__()
        self.device = torch.device("mps")
        self.norm1 = nn.LayerNorm(n_embed)
        self.multi_head = MulltiHeadAttention(
            n_heads, n_embed, head_size, block_size, dropout
        )
        self.norm2 = nn.LayerNorm(n_embed)
        self.ff_net = FFNetwork(n_embed, dropout)

    def forward(self, x):
        # We apply pre-norm instead of post-norm
        x = x + self.multi_head(self.norm1(x))
        x = x + self.ff_net(self.norm2(x))
        return x
