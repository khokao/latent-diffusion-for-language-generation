import torch.nn as nn
from omegaconf import ListConfig, OmegaConf

from .attention import Attention
from .feed_forward import FeedForward
from .positions import RelativePositionBias
from .residual import Residual, TimeConditionedResidual


class TransformerLayers(nn.Module):
    def __init__(
        self,
        in_dim,
        time_emb_dim,
        head_dim=64,
        heads=12,
        depth=12,
        layer_block=('a', 'c', 'f'),
        attn_dropout=0.1,
        ff_dropout=0.1,
        use_condition=True,
    ):
        super().__init__()
        self.use_condition = use_condition

        if isinstance(layer_block, ListConfig):
            layer_block = OmegaConf.to_object(layer_block)
        self.layer_types = layer_block * depth

        self.rel_pos = RelativePositionBias(scale=head_dim ** 0.5, heads=heads)

        self.layers = nn.ModuleList([])
        for layer_type in self.layer_types:
            if layer_type in ['a', 'c']:
                block = Attention(in_dim=in_dim, heads=heads, dropout=attn_dropout)
            elif layer_type == 'f':
                block = FeedForward(in_dim=in_dim, out_dim=in_dim, dropout=ff_dropout)
            else:
                raise ValueError(f'Unknown layer type {layer_type}')

            if layer_type == 'f':
                residual_fn = TimeConditionedResidual(time_emb_dim=time_emb_dim, out_dim=in_dim)
            else:
                residual_fn = Residual(dim=in_dim)

            norm_fn = nn.LayerNorm(in_dim)

            layer = nn.ModuleList([
                norm_fn,
                block,
                residual_fn,
            ])
            self.layers.append(layer)

    def forward(
        self,
        x,
        x_mask,
        time_emb,
        context=None,
        context_mask=None,
    ):
        assert self.use_condition == (context is not None)

        for layer_type, layer in zip(self.layer_types, self.layers):
            norm_fn, block, residual_fn = layer

            residual = x

            x = norm_fn(x)

            if layer_type == 'a':
                x = block(x, x_mask=x_mask, rel_pos=self.rel_pos)
            elif layer_type == 'c':
                x = block(x, x_mask=x_mask, context=context, context_mask=context_mask)
            elif layer_type == 'f':
                x = block(x)

            if layer_type == 'f':
                x = residual_fn(x, residual, time_emb)
            else:
                x = residual_fn(x, residual)

        return x
