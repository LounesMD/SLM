import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lorem_text import lorem

from bpe.tokenizer import Tokenizer
from model.trans_model import Transformer
from model.utils.utils import CharTokenizer, batch

# Train on mps
device = torch.device("mps")

# We gonna train and work with lorem ipsum
batch_size = 32
block_size = 8  # max_seq_length

text = "shakespeare"
if text == "lorem":
    text = lorem.paragraphs(10000)
elif text == "shakespeare":
    text = open("shakespeare.txt").read()
else:
    exit()

# We split our text into 90% train and 10% validation
train_data = text[: int(0.9 * len(text))]
val_data = text[int(0.9 * len(text)) :]

# Initialize the tokenizer
if False:
    # Load a trained tokenizer
    tok = Tokenizer()
    path = "./bpe/tokenizer/models"
    tok.load(path + ".model")
else:
    # Use this one for simple training
    tok = CharTokenizer(text)
# Save the model's weights
save = True
train_model = True

# nanoGPT parameters
n_head = 6
n_embd = 32
n_layer = 6
vocab_size = tok.vocab_size
training_iterations = 5000
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+

model = Transformer(
    vocab_size=vocab_size,
    n_embed=n_embd,
    n_heads=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
    training_iterations=training_iterations,
).to(device)

# Generation example
if False:
    print("##### Example of generation #####")
    msg = lorem.words(8)
    print("Input message: ", msg)
    (x, y) = batch(
        data=torch.tensor(tok.encode(msg), dtype=torch.long).to(device),
        block_size=block_size,
        batch_size=1,
        device=device,
    )
    print("Input sequence: ", tok.decode(x[0].detach().cpu().tolist()))
    res = model.generate(x, 10)
    print("Decoded output message: ", tok.decode(res[0].detach().cpu().tolist()))
    print("##### End of generation example #####")

print(model)

if train_model:
    model.fit(
        tok,
        training_data=tok.encode(train_data),
        block_size=block_size,
        batch_size=batch_size,
        eval_data=tok.encode(val_data),
    )
    if save:
        model.save("weights/model.pth")
else:
    model.load("weights/model.pth")

print("##### Example after Training #####")
x = torch.zeros((1, 1), dtype=torch.long).to(device)
res = model.generate(x, 100)
print("Decoded output message: ", tok.decode(res[0].detach().cpu().tolist()))
print("##### End of generation example #####")
