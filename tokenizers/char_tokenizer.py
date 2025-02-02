class CharTokenizer:
    def __init__(self, text) -> None:
        # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])
