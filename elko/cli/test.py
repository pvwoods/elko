import torch
from elko.transformer.models import FeedForwardMLP

x = torch.randn(3,10,26)
m = FeedForwardMLP(26,200)

print(m(x).shape)