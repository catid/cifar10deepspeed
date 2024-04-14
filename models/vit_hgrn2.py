# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import torch
import torch.nn.functional as F
from torch import nn

from typing import List, Optional, Tuple, Union

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .srmsnorm import FastSimpleRMSNorm
from .gla.recurrent_fuse import fused_recurrent_gla
from .gla.inter_chunk_contribution.fn import inter_chunk_onc
from .gla.intra_chunk_contribution.fn import intra_chunk_onc

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        y = self.net(x)
        return y

class Hgru2(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=2,
        bias=False,
    ):
        super().__init__()
        # get local varables
        params = locals()

        self.expand_ratio = expand_ratio
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_act = nn.GELU()
        self.out_act = nn.GELU()

        self.chunk_size = 128

    def forward(self, x, lower_bound=0.0):
        ## x: n b d
        n, b, d = x.shape
        feature = self.in_proj(x)
        V, Q, F_ = feature.chunk(3, dim=-1)
        V = self.in_act(V)
        Q = self.out_act(Q)
        F_ = F.sigmoid(F_)

        # reshape
        # h is num_head, d is head dimension
        V, Q, F_, lower_bound = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [V, Q, F_, lower_bound],
        )

        lambda_ = lower_bound + (1 - lower_bound) * F_

        log_lambda_ = torch.log(lambda_)

        K = 1 - lambda_

        if self.training:
            V, Q, G_K, K = map(
                lambda x: rearrange(
                    self.pad(x), "(n c) b h d -> b h n c d", c=self.chunk_size
                ).contiguous(),
                [V, Q, log_lambda_, K],
            )
            G_V = None
            G_K, G_V, o1 = inter_chunk_onc(Q, K, V, G_K, G_V)
            o2 = intra_chunk_onc(Q, K, V, G_K, G_V)
            o = o1 + o2
            o = rearrange(o, "b h n c d -> (n c) b (h d)")
        else:
            V, Q, G_K, K = map(
                lambda x: rearrange(x, "n b h d -> b h n d")
                .to(torch.float32)
                .contiguous(),
                [V, Q, log_lambda_, K],
            )
            o = fused_recurrent_gla(Q, K, V, G_K)
            o = rearrange(o, "b h n d -> n b (h d)").to(x.dtype)

        # out proj
        output = self.out_proj(o[:n])

        return output

    def pad(self, x):
        # n, b, h, d
        n, b, h, d = x.shape
        if n % self.chunk_size == 0:
            return x
        else:
            pad = self.chunk_size - n % self.chunk_size
            return F.pad(x, (0, 0, 0, 0, 0, 0, 0, pad)).contiguous()

class Transformer(nn.Module):
    def __init__(self, dim, depth, expand_ratio, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = FastSimpleRMSNorm(dim)

        self.lower_bounds = nn.Parameter(
            torch.ones(depth, dim), requires_grad=True
        )

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Hgru2(embed_dim=dim, expand_ratio=expand_ratio, bias=False),
                    FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                ])
            )
    def forward(self, x):
        # lower bound
        lower_bounds = self.lower_bounds
        lower_bounds = F.softmax(lower_bounds, dim=0)
        lower_bounds = torch.cumsum(lower_bounds, dim=0)
        lower_bounds -= lower_bounds[0, ...].clone()

        for idx, (attn, ff) in enumerate(self.layers):
            lower_bound = lower_bounds[idx]
            x = attn(self.norm(x), lower_bound) + x
            x = ff(self.norm(x)) + x
        x = self.norm(x)
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, expand_ratio, mlp_dim, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, expand_ratio, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
