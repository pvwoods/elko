import torch
import torch.nn as nn

class ResnetDistBlock(nn.Module):

    def __init__(self, in_dims:int, out_dims:int, groups:int):

        super(ResnetBlock, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.groups = groups

        self.projection = nn.Conv2d(self.in_dims, self.out_dims, 3, padding=1)
        self.activation = nn.SiLU()
        self.norm = nn.GroupNorm(self.groups, self.out_dims)

    def forward(self, x):
        h = self.projection(x)
        h = self.norm(h)
        return self.activation(h)

class ResnetBlock(nn.Module):

    def __init__(self, in_dims:int, out_dims:int, groups:int):

        super(ResnetBlock, self).__init__()

        self.in_block = ResnetDistBlock(in_dims, out_dims, groups)
        self.out_block = ResnetDistBlock(out_dims, in_dims, groups)
        self.residual_convolution = nn.Conv2d(in_dims, out_dims, 1) if in_dims != out_dims else nn.Identity()

    def forward(self, x):
        h = self.in_block(x)
        h = self.out_block(h)
        return h + self.residual_convolution(x)