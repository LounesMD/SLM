import unicodedata
from collections import Counter
from typing import List


#### Helper functions ####
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)  # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


###########################


class Tokenizer:
    def __init__(self, vocab_size=270) -> None:
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}  # We use utf-8, so we know the first 256 characters.
        self.current_idx = 256  # We need this parameter to perform the batch training.

        self.special_tokens = {}  # TODO: Implement special tokens.
        self.pattern = (
            ""  # TODO: Implement the pattern for splitting the text into tokens.
        )

    def pairs(self, words: List):
        """
        We assume that words is a List of tokens.
        """
        pairs = []
        prev = words[0]
        for char in words[1:]:
            pairs.append((prev, char))
            prev = char
        return pairs

    def merge(self, words, pair, nv_pair):
        """
        Merge the pair in the words.
        """
        new_words = []
        i = 0
        while i < len(words):
            if i < len(words) - 1 and (words[i], words[i + 1]) == pair:
                new_words.append(nv_pair)
                i += 2
            else:
                new_words.append(words[i])
                i += 1
        return new_words

    def bpe(self, bytes_text):
        """
        bpe training algorithm.
        bytes is a list where all characters have been split and encoded to bytes.
        """
        current_idx = 0
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        while True:
            pairs = self.pairs(bytes_text)
            if not pairs or current_idx >= (self.vocab_size - 256):
                break
            freq = self.get_stats(pairs)
            most_freq_pair = max(freq, key=freq.get)

            bytes_text = self.merge(bytes_text, most_freq_pair, 256 + current_idx)
            merges[most_freq_pair] = 256 + current_idx
            vocab[256 + current_idx] = (
                vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]
            )

            current_idx += 1

        self.merges = merges
        self.vocab = vocab

    def get_stats(self, freq_pairs):
        """
        Get the most frequent pair.
        """
        freq = {}
        for pair in freq_pairs:
            if pair in freq:
                freq[pair] += 1
            else:
                freq[pair] = 1
        return freq

    def decode(self, encoded_text):
        """
        The goal is to go from encoded_text utf-8 to text.
        """
        # We extend the vocab with the merges.
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        text_bytes = b"".join(self.vocab[idx] for idx in encoded_text)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        The goal is to go from text to encoded_text utf-8.
        """
        text_bytes = list(text.encode("utf-8"))
        # Encode using the trained merges.
        while len(text_bytes) > 1:
            pairs = self.pairs(text_bytes)
            stats = self.get_stats(pairs)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            text_bytes = self.merge(text_bytes, pair, self.merges[pair])

        return text_bytes

    def accumulate_stats(self, batch: List[str], freq: Counter):
        """Accumulate pair frequencies from a batch."""
        for text in batch:
            # tokens = list(text.encode("utf-8"))  # Convert to UTF-8 bytes
            pairs = self.pairs(text)
            freq.update(pairs)  # Update global frequency counter

    def apply_merges(self, batch: List[str]):
        """Apply the learned merges to the batch."""
        for i, text in enumerate(batch):
            tokens = list(text.encode("utf-8"))
            merged_tokens = self.encode(tokens)
            batch[i] = merged_tokens
        return batch

    def apply_all_merges(self, tokens: List[int]) -> List[int]:
        """Apply all merges in the order they were learned."""
        for pair, new_token in self.merges.items():
            tokens = self.merge(tokens, pair, new_token)
        return tokens

    def wrap_generator(self, corpus_generator, apply_all_merges):
        """Wrap the original generator to apply all merges before yielding data."""

        def wrapped_generator():
            for batch in corpus_generator():
                # Apply all learned merges to each document in the batch
                yield [apply_all_merges(list(doc.encode("utf-8"))) for doc in batch]

        return wrapped_generator

    def batch_train(self, corpus_generator):
        """Train the tokenizer using a corpus generator."""
        self.vocab = {
            idx: bytes([idx]) for idx in range(256)
        }  # Initialize with byte tokens
        wrapped_corpus_generator = self.wrap_generator(
            corpus_generator, self.apply_all_merges
        )

        num_merges = self.vocab_size - 256  # Number of merges to perform

        for idx in range(num_merges):
            print(f"Training merge {idx + 1}/{num_merges}")
            freq = Counter()

            # Step 1: Accumulate pair frequencies over all batches
            for batch in wrapped_corpus_generator():
                self.accumulate_stats(batch, freq)

            # Step 2: Perform merges based on the most frequent pairs
            if not freq:
                break

            # Find the most frequent pair
            most_freq_pair = max(freq, key=freq.get)
            new_token = self.current_idx
            self.merges[most_freq_pair] = new_token
            self.vocab[new_token] = (
                self.vocab[most_freq_pair[0]] + self.vocab[most_freq_pair[1]]
            )
            self.current_idx += 1
            # Apply the merge to each batch in-place
            for batch in wrapped_corpus_generator():
                for i in range(len(batch)):
                    # tokens = list(batch[i].encode("utf-8"))
                    batch[i] = self.merge(batch[i], most_freq_pair, new_token)

                # Recalculate frequencies after merging
                freq = Counter()
                for batch in wrapped_corpus_generator():
                    self.accumulate_stats(batch, freq)
            print(f"Most frequent pair: {most_freq_pair} -> {new_token}")

    def _build_vocab(self):
        """
        Copy paste from: https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
        """
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def load(self, model_file):
        """
        Copy paste from: https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
        Inverse of save() but only for the model file
        """
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

    def save(self, file_prefix):
        """
        Copy paste from: https://github.com/karpathy/minbpe/blob/master/minbpe/base.py
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")
