import torch
from torch.utils.data import Dataset
from elko.transformer.tokenizer import ascii_tokenizer

class AsciiTokenizedTextFileDataset(Dataset):
    def __init__(self, file_path:str, block_size:str):

        self.PADDING_TOKEN = 10

        self.file_path = file_path
        self.block_size = block_size

        text = open(self.file_path).read()
        tokenized_text = ascii_tokenizer(text)

        # pad the end to be divisible by block size
        padding_needed = self.block_size - (len(tokenized_text) % self.block_size)
        padding = [self.PADDING_TOKEN] * padding_needed
        tokenized_text = tokenized_text + padding

        # convert to tensor
        self.data = torch.tensor(tokenized_text).view(-1, self.block_size)
    

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]