import torch
from typing import Any

class BasicCharTokenizer:

    def __init__(self, text) -> None:
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.token_lookup = {ch:i for i,ch in enumerate(self.chars)}
        self.char_lookup = {i:ch for i,ch in enumerate(self.chars)}
    
    def encode(self, s):
        return torch.tensor([self.token_lookup[c] for c in s])

    def decode(self, t):
        return ''.join([self.char_lookup[i] for i in t])

    def __call__(self, x) -> Any:
        if isinstance(x, str):
            return self.encode(x)
        else:
            return self.decode(x)