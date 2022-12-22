import torch
import random
from elko.transformer.models import CausalTransformer
from elko.datasets.text import AsciiTokenizedTextFileDataset
import sys
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


def build_dataset(
    corpus, block_size, codebook, start_padding_char="#", end_padding_char="$"
):
    X, Y = [], []
    for word in corpus:
        word_len = len(word)
        end_padding_amount = block_size - (word_len + 1)
        end_padding = end_padding_char * end_padding_amount
        padded_word = f"{start_padding_char}{word}{end_padding}"
        tokenized_word = [codebook[c] for c in padded_word]
        X.append(tokenized_word[: block_size - 1])
        Y.append(tokenized_word[1:])

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return {"x": X, "y": Y}


def build_token_dict(words):

    chars = sorted(list(set("".join(words))))  # lol
    token_lookup = {c: i + 1 for i, c in enumerate(chars)}
    token_lookup["#"] = 0
    token_lookup["$"] = len(chars) + 1
    char_lookup = {i: c for c, i in token_lookup.items()}
    total_tokens = len(char_lookup.keys())

    return token_lookup, char_lookup, total_tokens


def build_simple_dataset_from_file(file_path):

    words = open(file_path).read().splitlines()

    token_lookup, char_lookup, vocab_size = build_token_dict(words)

    # build the dataset

    block_size = max([len(w) for w in words]) + 2

    random.shuffle(words)

    n1 = int(len(words) * 0.8)
    n2 = int(len(words) * 0.9)

    return {
        "train": build_dataset(words[:n1], block_size, token_lookup),
        "test": build_dataset(words[n1:n2], block_size, token_lookup),
        "valid": build_dataset(words[n2:], block_size, token_lookup),
        "vocab_size": vocab_size,
        "block_size": block_size,
        "token_lookup": token_lookup,
        "char_lookup": char_lookup,
    }


def train(model, dataloader, batch_size=16, epochs=20, device="cuda:1"):

    optim = torch.optim.Adam(model.parameters())
    pbar = tqdm(range(len(dataloader) * epochs))

    losses = []

    model.to(device)

    for _ in pbar:
        data_iter = iter(dataloader)
        for batch in data_iter:
            x_batch = batch[:,:-1].to(device)
            logits = model(x_batch)
            y = batch[:,1:].contiguous().view(-1)

            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.to(device))
            losses.append(loss.item())
            pbar.set_description(desc=f"average loss: {(sum(losses[-100:])/100):.2f}")

            model.zero_grad()
            loss.backward()
            optim.step()


if __name__ == "__main__":
    
    dataset = AsciiTokenizedTextFileDataset(sys.argv[1], 64)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CausalTransformer(
        vocab_size=128,
        num_layers=2,
        embedding_dims=64,
        heads=4,
        block_size=64,
        ff_dims=256,
    )

    print(f"model size is {sum(p.numel() for p in model.parameters())}")
    

    train(model, dataloader)
    

