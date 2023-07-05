import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(
        self,
        in_dim,
        qk_head_dim=64,
        v_head_dim=64,
        heads=12,
        dropout=0.1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = qk_head_dim ** -0.5

        q_dim = qk_head_dim * self.heads
        k_dim = q_dim
        v_dim = v_head_dim * self.heads

        self.to_q = nn.Linear(in_dim, q_dim, bias=False)
        self.to_k = nn.Linear(in_dim, k_dim, bias=False)
        self.to_v = nn.Linear(in_dim, v_dim, bias=False)
        self.to_out = nn.Linear(v_dim, in_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        x_mask,
        context=None,
        context_mask=None,
        rel_pos=None,
    ):

        with_context = bool(context is not None)
        kv_input = context if with_context else x

        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        qk_dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        mask_value = - torch.finfo(qk_dots.dtype).max

        if rel_pos is not None:
            qk_dots = rel_pos(qk_dots)

        q_mask = x_mask
        k_mask = context_mask if with_context else x_mask

        q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
        k_mask = rearrange(k_mask, 'b j -> b 1 1 j')

        input_mask = (q_mask * k_mask).bool()
        qk_dots.masked_fill_(~input_mask, mask_value)
        del input_mask

        attn = qk_dots.softmax(dim=-1, dtype=torch.float32)
        attn = self.dropout(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out
