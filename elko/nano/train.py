import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .tokenizer import BasicCharTokenizer
from .models import BigramLanguageModel

# hyperparameters

BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 1000
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200

torch.manual_seed(1337)

def load_data(path:str):
    with open(path) as file:
        return file.read()

def build_batch_generator(train_data, val_data):

    def get_batch(split:str="train"):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x, y
    
    return get_batch

def get_data_utils(file_path:str):
    
    text = load_data(file_path)

    tokenizer = BasicCharTokenizer(text)

    data = tokenizer(text)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return tokenizer, build_batch_generator(train_data, val_data)


def train(model, tokenizer, optimizer, get_batch_fn):

    for step in range(MAX_ITERS):

        xb, yb = get_batch_fn()

        out, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % EVAL_INTERVAL == 0:
            print(loss.item())
            out = model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)
            print(tokenizer(out[0].tolist()))




if __name__ == '__main__':

    text_file = sys.argv[1]
    

    tokenizer, get_batch_fn = get_data_utils(text_file)
    model = BigramLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train(model, tokenizer, optimizer, get_batch_fn)