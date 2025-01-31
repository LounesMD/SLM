from typing import List 
import unicodedata


#### Helper functions ####
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s
###########################

class tokenizer:
    def __init__(self, vocab_size = 270) -> None:
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {} # We use utf-8, so we know the first 256 characters.
        
        self.special_tokens = {} # TODO: Implement special tokens.
        self.pattern = "" # TODO: Implement the pattern for splitting the text into tokens.

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
            if i < len(words) - 1 and (words[i], words[i+1]) == pair:
                new_words.append(nv_pair)
                i += 2
            else:
                new_words.append(words[i])
                i += 1
        return new_words

    def bpe(self, bytes):
        """
        bpe training algorithm.
        bytes is a list where all characters have been splitted and encoded to bytes.
        """
        current_idx = 0
        merges = {}
        vocab = {i: bytes[i] for i in range(256)}

        while True:           
            pairs = self.pairs(bytes)
            if not pairs or current_idx >= (self.vocab_size - 256):
                break
            freq = self.get_stats(pairs)
            most_freq_pair = max(freq, key=freq.get)

            bytes = self.merge(bytes, most_freq_pair, 256 + current_idx)
            merges[most_freq_pair] = 256 + current_idx
            vocab[256 + current_idx] = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]

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
            self.vocab[idx] = pair[0] + pair[1]
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
        with open(model_file, 'w') as f:
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
                breakpoint()
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
