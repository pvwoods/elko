import torch
import random
from elko.transformer.models import CausalTransformer
from elko.datasets.text import NamesDataset
import sys
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


@torch.no_grad()
def p_sample(model, tokens, temp=1.0, top_k=3):
    pos = len(tokens)
    tokens = tokens + [0] * (model.block_size - pos)
    while True:
        cur_tokens = torch.tensor(tokens).unsqueeze(0)
        logits = model(cur_tokens)
        logits = logits[:, pos, :] / temp
        top_k_vals, _ = torch.topk(logits, top_k)
        # set any value less than the K'th value to -inf
        kth_vals = top_k_vals[:, [-1]]
        logits[logits < kth_vals] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        tokens[pos + 1] = ix
        pos += 1
        if ix == 27 or pos == model.block_size - 1:
            break
    decoded = "".join([chr((t + ord('a')) - 1) for t in tokens])
    return decoded.split('{')[0][1:]

def train(model, dataloader, epochs=5, device="cuda:1"):

    optim = torch.optim.Adam(model.parameters())
    pbar = tqdm(range(epochs))

    losses = []

    model.to(device)

    for _ in pbar:
        data_iter = iter(dataloader)
        for batch, targets in data_iter:
            batch = batch.to(device)
            logits = model(batch)
            targets = targets.view(-1).to(device)

            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets)
            losses.append(loss.item())
            pbar.set_description(desc=f"average loss: {(sum(losses[-100:])/100):.2f}")

            model.zero_grad()
            loss.backward()
            optim.step()
        print(f'samples:', [p_sample(model, []) for _ in range(5)])

if __name__ == "__main__":

    dataset = NamesDataset(sys.argv[1], 32)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CausalTransformer(
        vocab_size=28,
        num_layers=3,
        embedding_dims=16,
        heads=4,
        block_size=32,
        ff_dims=64,
    )

    print(f"model size is {sum(p.numel() for p in model.parameters())}")

    train(model, dataloader, device='cpu', epochs=3)
