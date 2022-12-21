import torch
from elko.transformer.models import FeedForwardMLP, CausalSelfAttention

x = torch.randn(3, 10, 256)
m = FeedForwardMLP(256, 200)
a = CausalSelfAttention(256, 8, 64)

print(m(x).shape)
print(a(x).shape)
