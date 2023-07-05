import torch.nn as nn

from .utils import init_zero_


class GeGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.act = nn.GELU()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        x = x * self.act(gate)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, mult=4, init_zero_last=False):
        super().__init__()
        inner_dim = int(in_dim * mult)
        self.ff = nn.Sequential(
            GeGLU(in_dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, out_dim),
        )

        if init_zero_last:
            init_zero_(self.ff[-1])

    def forward(self, x):
        out = self.ff(x)
        return out
