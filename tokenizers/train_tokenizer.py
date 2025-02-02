import os
from collections import Counter
from typing import List

from datasets import load_dataset
from lorem_text import lorem

from tokenizers.bpe_tokenizer import Tokenizer

raw_datasets = load_dataset("code_search_net", "python")
path = "./bpe/tokenizer/models"
os.makedirs(path, exist_ok=True)


def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
        if start_idx > 5000:
            break


# Initialize the tokenizer
tokenizer = Tokenizer(vocab_size=280)  # Example: vocab size of 1000 tokens
train = True
# Train on the corpus
if train:
    tokenizer.batch_train(get_training_corpus)
else:
    tokenizer.load(path + ".model")

# Test encoding and decoding
test_text = lorem.paragraph()
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

tokenizer.save(path)
print("utf text:", list(test_text.encode("utf-8")))
print("Encoded:", encoded)
print("Decoded:", decoded)
print("Compression ratio:", len(list(test_text.encode("utf-8"))) / len(encoded))
