import torch
from torch.utils.data import Dataset
from typing import Callable

class TokenizedTextFileDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: str,
        pre_processing_fn: Callable,
        tokenizer_fn: Callable
    ):

        self.PADDING_TOKEN = 10

        self.file_path = file_path
        self.block_size = block_size
        self.pre_processing_fn = pre_processing_fn
        self.tokenizer_fn = tokenizer_fn

        text = open(self.file_path).read()

        processed_text = self.pre_processing_fn(text)

        tokenized_text = self.tokenizer_fn(processed_text)

        # convert to tensor
        data = torch.tensor(tokenized_text).view(-1, self.block_size)
        self.inputs = data[:, :-1]
        self.targets = data[:, 1:].contiguous()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def name_tokenizer(x: str, block_size:int):
    """
    a very simple tokenizer, assumes that these are names composed
    of the 26 characters of the alphabet
    """
    name_tokens = [0] + [(ord(xi) - ord("a")) + 1 for xi in x.lower().strip()]
    padding_needed = block_size - len(name_tokens)
    padding = [27] * padding_needed
    return name_tokens + padding

class NamesDataset(TokenizedTextFileDataset):

    def __init__(self, file_path:str, block_size:int):
        
        # TODO: I need to make the sliding window for this dataset
        preprocess = lambda text: text.split('\n')
        tokenizer = lambda xs: [name_tokenizer(x, block_size) for x in xs]

        super(NamesDataset, self).__init__(file_path, block_size, preprocess, tokenizer)