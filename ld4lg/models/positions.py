"""
The codes are modified.

Link:
    - [SinusoidalPossitionalEmbedding] https://github.com/ermongroup/ddim/
      blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/models/diffusion.py#L6-L24
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # NOQA
from einops import rearrange


class SinusoidalPossitionalEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Args:
            dim (int): Number of embedded dimensions.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Embedded values.
                shape = (size, )
                dtype = torch.float32

        Returns:
            emb (toch.tensor): Sinusoidal embeddings.
                shape = (size, dim)
                dtype = torch.float32
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, use_l2norm=False):
        super().__init__()
        self.scale = dim ** -0.5 if not use_l2norm else 1.
        self.max_seq_len = max_seq_len
        self.use_l2norm = use_l2norm
        self.emb = nn.Embedding(max_seq_len, dim)
        self.l2norm = L2Normalization(groups=1) if use_l2norm else None

    def forward(self, x, pos=None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'Sequence length {seq_len} exceeds max {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        pos_emb = self.l2norm(pos_emb) if self.use_l2norm else pos_emb
        return pos_emb


class L2Normalization(nn.Module):
    def __init__(self, groups=1):
        super().__init__()
        self.groups = groups

    def forward(self, t):
        t = rearrange(t, '... (g d) -> ... g d', g=self.groups)
        t = F.normalize(t, p=2, dim=-1)
        t = rearrange(t, '... g d -> ... (g d)')
        return t


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)

        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return qk_dots + (bias * self.scale)
