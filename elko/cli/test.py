import torch
import random
from elko.transformer.models import CausalTransformer
from elko.datasets.text import NamesDataset
import sys
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


@torch.no_grad()
def p_sample(model, tokens, temp=1, top_k=0):
    pos = max(0, len(tokens) - 1)
    tokens = tokens + [28] * (model.block_size - len(tokens))
    while True:
        cur_tokens = torch.tensor(tokens).unsqueeze(0)
        logits = model(cur_tokens).squeeze(0)[pos]
        if temp != 1:
            logits /= temp
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, top_k)
            # set any value less than the K'th value to -inf
            kth_vals = top_k_vals[:, [-1]]
            logits[logits < kth_vals] = float('-inf')
        probs = torch.softmax(logits, dim=0)
        ix = torch.multinomial(probs, num_samples=1).item()
        tokens[pos + 1] = ix
        pos += 1
        if ix == 27 or pos == model.block_size - 1:
            break
    decoded = "".join([chr((t + ord('a')) - 1) for t in tokens])
    return decoded.split('{')[0][1:]

def train(model, dataloader, epochs=5, device="cuda:1"):

    optim = torch.optim.Adam(model.parameters())
    pbar = tqdm(total=epochs * len(dataloader))

    losses = []

    model.to(device)

    for e in range(epochs):
        data_iter = iter(dataloader)
        for batch, targets in data_iter:
            pbar.update(1)
            batch = batch.to(device)
            logits = model(batch)
            targets = targets.view(-1).to(device)

            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets, ignore_index=28)
            losses.append(loss.item())
            pbar.set_description(desc=f"Epoch {e}/{epochs} | average loss: {(sum(losses[-100:])/min(100,len(losses))):.2f}")

            model.zero_grad()
            loss.backward()
            optim.step()
        print(f'unconditional samples:', [p_sample(model, []) for _ in range(5)])
    print(f'Final conditioned Samples:', [p_sample(model, [0, i]) for i in range(1, 27)])

if __name__ == "__main__":

    dataset = NamesDataset(sys.argv[1], 20)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = CausalTransformer(
        vocab_size=29,
        num_layers=4,
        embedding_dims=32,
        heads=4,
        block_size=20,
        ff_dims=128,
    )

    print(f"model size is {sum(p.numel() for p in model.parameters())}")

    train(model, dataloader, device='cpu', epochs=5)
