import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.trans_decoder import TransformerDecoder
from model.trans_encoder import TransformerEncoder
from model.utils.utils import batch, batch_translation, pad_sequence, pos_embeddings


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
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, y=None, full_architecture=False, compute_loss=True):
        x = self.token_embedding(x)
        x += pos_embeddings(x.shape[1], x.shape[2], device=self.device).unsqueeze(0)
        # # Encoder forward
        if full_architecture:
            # In case of Encoder-Decoder, we have two inputs for the Decoder: One for the Decoder input and one for the Encoder output (key-value pairs)
            enc_output = self.encoder(x)
            # The input of the decoder is the target sequence but shifted by one token to the right
            if y is None:
                dec_input = torch.full(
                    (x.size(0), 1), fill_value=100, dtype=torch.long, device=self.device
                )
            else:
                dec_input = y
            sos_token = torch.full(
                (dec_input.size(0), 1),
                fill_value=100,
                dtype=torch.long,
                device=self.device,
            )
            dec_input = torch.cat([sos_token, dec_input[:, :-1]], dim=1)
            dec_input = self.token_embedding(dec_input)
            dec_input += pos_embeddings(
                dec_input.shape[1], dec_input.shape[2], device=self.device
            ).unsqueeze(0)
            x = self.decoder((dec_input, enc_output))
        else:
            # In the case of Decoder only, there is only one input
            x = self.decoder(x)
        # Logits prediction
        logits = self.proj_linear(x)
        if y is None:
            loss = None
        else:
            if compute_loss:
                # TODO: Check the eos token for the loss
                B, T, C = logits.size()
                logits = logits.reshape(B * T, C)
                y = y.reshape(B * T)

                loss = F.cross_entropy(logits, y)
            else:
                loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, x, gen_length):
        for _ in range(gen_length):
            logits, _ = self.forward(x[:, -self.block_size :], y=None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=-1)
        return x

    @torch.no_grad()
    def translate(self, message, max_gen_length=16, sos_token=100, eos_token=101):
        """
        Translates a source message using the trained encoder-decoder model.
        """
        source = message.to(self.device).unsqueeze(0)
        dec_input = torch.tensor([], dtype=torch.long, device=self.device)
        for _ in range(max_gen_length):
            logits, _ = self.forward(
                source,
                y=dec_input.unsqueeze(0),
                full_architecture=True,
                compute_loss=False,
            )
            logits = logits[-1]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)[-1]
            dec_input = torch.cat([dec_input, next_token], dim=0)
            if next_token.item() == eos_token:
                break
        return dec_input

    @torch.no_grad()
    def get_loss(self, eval_data, block_size, batch_size, idx, task):
        # Evaluation
        self.eval()
        if task == "generation":
            (x, y) = batch(
                data=torch.tensor(eval_data, dtype=torch.long).to(self.device),
                block_size=block_size,
                batch_size=batch_size,
                device=self.device,
            )
            _, loss = self.forward(x, y, full_architecture=False)
        elif task == "translation":
            (x, y) = batch_translation(
                data_x=[
                    torch.tensor(elt, dtype=torch.long).to(self.device)
                    for elt in eval_data[0]
                ],
                data_y=[
                    torch.tensor(elt, dtype=torch.long).to(self.device)
                    for elt in eval_data[1]
                ],
                block_size=block_size,
                batch_size=batch_size,
                device=self.device,
            )
            _, loss = self.forward(x, y, full_architecture=True)
        print("Step {} Evaluation Loss: {}".format(idx, loss.item()))
        self.train()

    def fit_generation(self, training_data, block_size, batch_size, eval_data):
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
                self.get_loss(eval_data, block_size, batch_size, idx, task="generation")

    def fit_translation(self, training_data, block_size, batch_size, eval_data):
        for idx in range(self.training_iterations):
            (x, y) = batch_translation(
                data_x=[
                    torch.tensor(elt, dtype=torch.long).to(self.device)
                    for elt in training_data[0]
                ],
                data_y=[
                    torch.tensor(elt, dtype=torch.long).to(self.device)
                    for elt in training_data[1]
                ],
                block_size=block_size,
                batch_size=batch_size,
                device=self.device,
            )
            # 1. Token Embedding + Encoder forward + Decoder forward
            _, loss = self.forward(x, y, full_architecture=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if idx % 50 == 0:
                self.get_loss(
                    eval_data, block_size, batch_size, idx, task="translation"
                )
                print("Step {} training Loss: {}".format(idx, loss.item()))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
