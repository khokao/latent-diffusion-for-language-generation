import torch
import torch.nn as nn
from einops import rearrange, repeat

from .positions import AbsolutePositionalEmbedding, SinusoidalPossitionalEmbedding
from .transformer_layers import TransformerLayers


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        x_dim,
        seq_len,
        hidden_dim,
        head_dim=64,
        heads=12,
        depth=12,
        layer_block=('a', 'c', 'f'),
        attn_dropout=0.1,
        ff_dropout=0.1,
        use_self_condition=True,
        use_class_condition=True,
        num_classes=None,
    ):
        super().__init__()

        self.use_self_condition = use_self_condition
        self.use_class_condition = use_class_condition
        self.use_condition = use_self_condition or use_class_condition

        self.input_proj = nn.Linear(x_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, x_dim)

        time_emb_dim = x_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPossitionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_pos_emb_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, hidden_dim),
        )

        self.pos_emb = AbsolutePositionalEmbedding(hidden_dim, seq_len)

        self.transformer_layers = TransformerLayers(
            in_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            head_dim=head_dim,
            heads=heads,
            depth=depth,
            layer_block=layer_block,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_condition=self.use_condition,
        )

        if self.use_self_condition:
            self.null_emb = nn.Embedding(1, hidden_dim)
            self.self_proj = nn.Linear(x_dim, hidden_dim)

        if self.use_class_condition:
            assert num_classes > 0, 'Must specify number of classes'
            self.class_emb = nn.Embedding(num_classes + 1, hidden_dim)

    def forward(self, x, x_mask, t, x_self_cond=None, class_id=None):
        b = x.shape[0]

        time_emb = self.time_mlp(t)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        h = self.input_proj(x)
        h += self.pos_emb(x)
        h += self.time_pos_emb_mlp(time_emb)

        context = None
        context_mask = None
        if self.use_condition:
            context = []
            context_mask = []

            if self.use_self_condition:
                if x_self_cond is None:
                    null_emb = repeat(self.null_emb.weight, '1 d -> b 1 d', b=b)
                    true_mask = torch.full((b, 1), True, device=x.device)

                    context.append(null_emb)
                    context_mask.append(true_mask)
                else:
                    self_emb = self.self_proj(x_self_cond)

                    context.append(self_emb)
                    context_mask.append(x_mask)

            if self.use_class_condition:
                assert class_id is not None, 'Must specify class id'
                class_emb = self.class_emb(class_id)
                class_emb = rearrange(class_emb, 'b d -> b 1 d')
                true_mask = torch.full((b, 1), True, device=x.device)

                context.append(class_emb)
                context_mask.append(true_mask)

            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)

        h = self.transformer_layers(h, x_mask=x_mask, time_emb=time_emb, context=context, context_mask=context_mask)

        h = self.norm(h)

        out = self.output_proj(h)

        return out
