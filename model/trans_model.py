import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.trans_decoder import TransformerDecoder
from model.trans_encoder import TransformerEncoder
from model.utils.utils import batch, batch_translation, pos_embeddings


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

    def forward(self, x, y=None, full_architecture=False):
        x = self.token_embedding(x)
        x += pos_embeddings(x.shape[1], x.shape[2], device=self.device).unsqueeze(0)
        # # Encoder forward
        if full_architecture:
            # In case of Encoder-Decoder, we have two inputs for the Decoder: One for the Decoder input and one for the Encoder output (key-value pairs)
            enc_output = self.encoder(x)
            # The input of the decoder is the target sequence but shifted by one token to the right
            if y is None:
                y = torch.full(
                    (x.size(0), 1), fill_value=100, dtype=torch.long, device=self.device
                )
            sos_token = torch.full(
                (y.size(0), 1), fill_value=100, dtype=torch.long, device=self.device
            )
            dec_input = torch.cat([sos_token, y[:, :-1]], dim=1)

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
            B, T, C = logits.size()
            logits = logits.reshape(B * T, C)
            y = y.reshape(B * T)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    @torch.no_grad()
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
    def translate(self, message, max_gen_length=50, sos_token=100, eos_token=101):
        """
        Translates a source message using the trained encoder-decoder model.
        """
        source = message.to(self.device).unsqueeze(0)
        dec_input = torch.tensor([[sos_token]], dtype=torch.long, device=self.device)
        # ---- Autoregressive generation loop ----
        for _ in range(max_gen_length):
            # Pass the source and current decoder input to the forward method.
            # We expect our forward method to accept a dec_input parameter (for the decoder input)
            # and use the encoder output to generate target tokens.
            logits, _ = self.forward(source, y=None, full_architecture=True)
            # logits shape is (1, T, vocab_size); take the logits for the last token.
            # next_logits = logits[:, -1, :]  # shape: (1, vocab_size)
            # Apply softmax to get probabilities (if not already applied in your proj_linear)
            probs = F.softmax(logits, dim=-1)
            # Sample the next token from the probability distribution.
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)
            # Append the next token to the decoder input sequence.
            dec_input = torch.cat([dec_input, next_token], dim=1)  # now shape: (1, T+1)

            # Optionally, break if the end-of-sequence token is generated.
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
        _, loss = self.forward(x, y)
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
            print(idx)
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
                # print(tok.decode(self.generate(torch.zeros((1, 1), dtype=torch.long).to(self.device),50)[0].detach().cpu().tolist()))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
