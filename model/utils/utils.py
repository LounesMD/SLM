import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """
    Single attention head
    """

    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.W_Q = nn.Linear(n_embed, head_size, bias=False)
        self.W_K = nn.Linear(n_embed, head_size, bias=False)
        self.W_V = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=False, enc_out=None):
        """
        Compute scaled dot-product attention.
        enc_out is used in the case of an Encoder-Decoder architecture. Key-Value of the MHA of the Encoder.
        """
        Q = self.W_Q(x)
        if enc_out is not None:
            K = self.W_K(enc_out)
            V = self.W_V(enc_out)
        else:
            K = self.W_K(x)
            V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        res = F.softmax(scores, dim=-1)
        if mask:
            res = res.masked_fill(
                self.tril[: x.shape[1], : x.shape[1]] == 0, float("-inf")
            )
            res = F.softmax(res, dim=-1)
        res = self.dropout(res)  # ?
        res = torch.matmul(res, V)

        return res


class MulltiHeadAttention(nn.Module):
    """
    Multi head attention
    """

    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(n_embed, head_size, block_size, dropout)
                for _ in range(n_heads)
            ]
        )
        self.fc_out = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out=None):
        # enc_out is used in the case of an Encoder-Decoder architecture. Key-Value of the MHA of the Decoder.
        x = torch.cat(
            [head(x, enc_out) for head in self.heads], dim=-1
        )  # Compute h times the attention
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


class MaskedMulltiHeadAttention(MulltiHeadAttention):
    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        head_size: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__(n_heads, n_embed, head_size, block_size, dropout)

    def forward(self, x):
        x = torch.cat(
            [head(x, mask=True) for head in self.heads], dim=-1
        )  # Compute h times the attention
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


class FFNetwork(nn.Module):
    """
    Feed forward network
    """

    def __init__(self, n_embed: int, dropout: float) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        # When using this very simple model, we basically predict the next token of each token individually
        self.predictor = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets):
        # x has shape (B,S) and after the embedding it has shape (B,S,C)
        x = self.predictor(x)
        B, T, C = x.size()
        x = x.reshape(B * T, C)
        loss = F.cross_entropy(x, targets.reshape(B * T))

        return x, loss


def batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def pad_sequence(seq, max_length, padding_value=99):
    if len(seq) < max_length:
        pad_size = max_length - len(seq)
        return torch.cat([seq, torch.full((pad_size,), padding_value).to(device="mps")])
    else:
        return seq[:max_length]  # Truncate if longer than max_length


def batch_translation(data_x, data_y, block_size, batch_size, device):
    assert len(data_x) == len(data_y)
    max_length = 16
    ix = torch.randint(len(data_x) - block_size, (batch_size,))
    x = torch.stack([pad_sequence(data_x[i], max_length) for i in ix]).to(device)
    y = torch.stack([pad_sequence(data_y[i], max_length) for i in ix]).to(device)
    x, y = x.to(device), y.to(device)
    return x, y


def pos_embeddings(seq_length, d_model, device):
    # Positional encoding
    pe = torch.empty(seq_length, d_model, device=device)
    pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000) / d_model)
    )
    # Even indices are encoded using the sin and Odd ones using the cos
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

    return pe


# TODO: Add the cross attention between encoder and decoder
