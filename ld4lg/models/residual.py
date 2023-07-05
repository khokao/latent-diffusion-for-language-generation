import torch
import torch.nn as nn
from einops import rearrange

from .utils import init_zero_


class Residual(nn.Module):
    def __init__(self, dim, apply_residual_scaling=False, residual_scale_constant=1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if apply_residual_scaling else None
        self.residual_scale_constant = residual_scale_constant

    def forward(self, x, residual):
        if self.residual_scale is not None:
            residual = residual * self.residual_scale

        if self.residual_scale_constant != 1:
            residual = residual * self.residual_scale_constant

        out = x + residual

        return out


class GRUResidual(nn.Module):
    def __init__(self, dim, apply_residual_scaling=False):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if apply_residual_scaling else None
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        if self.residual_scale is not None:
            residual = residual * self.residual_scale

        out = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )
        out = out.reshape_as(x)

        return out


class TimeConditionedResidual(nn.Module):
    def __init__(self, time_emb_dim, out_dim, init_zero_last=True):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_dim * 2)
        )

        if init_zero_last:
            init_zero_(self.time_mlp[-1])

    def forward(self, x, residual, time_emb):
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=2)
        x = x * (scale + 1) + shift
        out = x + residual

        return out
