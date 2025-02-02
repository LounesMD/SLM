import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.trans_decoder import TransformerDecoder
from model.trans_encoder import TransformerEncoder
from model.utils.utils import batch, pos_embeddings


class Transformer(nn.Module):
    """
    Transformer architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        block_size: int,
        n_layer: int,
        dropout: float,
        training_iterations: int,
    ):
        super(Transformer, self).__init__()
        head_size = n_embed // n_heads
        self.block_size = block_size
        self.device = torch.device("mps")
        self.training_iterations = training_iterations
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    n_heads=n_heads,
                    n_embed=n_embed,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.decoder = nn.Sequential(
            *[
                TransformerDecoder(
                    n_heads=n_heads,
                    n_embed=n_embed,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.proj_linear = nn.Sequential(
            nn.Linear(n_embed, vocab_size),
            # nn.Softmax(dim=-1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, y=None):
        x = self.token_embedding(x)
        x += pos_embeddings(x.shape[1], x.shape[2], device=self.device).unsqueeze(0)
        # # Encoder forward
        # TODO: Add the encoder as a parameter
        # x = self.encoder(x)
        # Decoder forward
        x = self.decoder(x)
        # Logits prediction
        logits = self.proj_linear(x)
        if y is None:
            loss = None
        else:
            B, T, C = logits.size()
            logits = logits.reshape(B * T, C)
            y = y.reshape(B * T)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, x, gen_length):
        for _ in range(gen_length):
            logits, _ = self.forward(x[:, -self.block_size :], y=None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # We sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat(
                [x, next_token], dim=-1
            )  # We append the next token to the sequence
        return x

    @torch.no_grad()
    def get_loss(self, eval_data, block_size, batch_size, idx):
        # Evaluation
        self.eval()
        (x, y) = batch(
            data=torch.tensor(eval_data, dtype=torch.long).to(self.device),
            block_size=block_size,
            batch_size=batch_size,
            device=self.device,
        )
        _, loss = self.forward(x, y)
        print("Step {} Evaluation Loss: {}".format(idx, loss.item()))
        self.train()

    def fit(self, tok, training_data, block_size, batch_size, eval_data):
        for idx in range(self.training_iterations):
            (x, y) = batch(
                data=torch.tensor(training_data, dtype=torch.long).to(self.device),
                block_size=block_size,
                batch_size=batch_size,
                device=self.device,
            )
            # 1. Token Embedding + Encoder forward + Decoder forward
            _, loss = self.forward(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if idx % 100 == 0:
                self.get_loss(eval_data, block_size, batch_size, idx)
                # print(tok.decode(self.generate(torch.zeros((1, 1), dtype=torch.long).to(self.device),50)[0].detach().cpu().tolist()))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
