# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .srmsnorm import FastSimpleRMSNorm

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

use_gqa = False

if use_gqa:

    class LSA(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super().__init__()
            self.dropout = dropout
            self.heads_q = heads
            self.heads_kv = heads // 2
            inner_dim_q = dim_head * self.heads_q
            inner_dim_kv = dim_head * self.heads_kv

            self.to_q = nn.Linear(dim, inner_dim_q, bias = False)
            self.to_k = nn.Linear(dim, inner_dim_kv, bias = False)
            self.to_v = nn.Linear(dim, inner_dim_kv, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim_q, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads_q)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads_kv)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads_kv)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=False,
                scale=None)

            out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads_q)
            return self.to_out(out)

else:

    class LSA(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super().__init__()
            self.dropout = dropout
            self.heads = heads
            inner_dim = dim_head * heads

            self.to_q = nn.Linear(dim, inner_dim, bias = False)
            self.to_k = nn.Linear(dim, inner_dim, bias = False)
            self.to_v = nn.Linear(dim, inner_dim, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)

            q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
            k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
            v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout,
                is_causal=False,
                scale=None)

            out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
            return self.to_out(out)

class SGLU(nn.Module):
    def __init__(self, d_in=64, mlp_dim=256, d_out=64, bias=False):
        super().__init__()

        self.in_u = nn.Linear(d_in, mlp_dim, bias=bias)
        self.in_v = nn.Linear(d_in, mlp_dim, bias=bias)
        self.out_proj = nn.Linear(mlp_dim, d_out, bias=bias)

    def forward(self, x):
        return self.out_proj(self.in_u(x) * self.in_v(x))

# Non-linear QKV attention experiment from https://arxiv.org/pdf/2111.10017.pdf
class LSA(nn.Module):
    def __init__(self, dim, mlp_dim = 256, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_qkv = SGLU(dim, mlp_dim, inner_dim * 3, bias = False)
        self.norm = FastSimpleRMSNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.norm(self.to_qkv(x))

        qkv = rearrange(qkv, 'b n (r h d) -> b r h n d', r=3, h=self.heads)
        q, k, v = torch.split(qkv, 1, dim=1)

        q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False,
            scale=None)

        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        return self.to_out(out)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        # Modified Attention Block from https://arxiv.org/pdf/2312.01324.pdf
        #self.act = nn.GELU()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Modified Attention Block from https://arxiv.org/pdf/2312.01324.pdf
        #v = self.act(self.to_v(x))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False,
            scale=None)

        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = FastSimpleRMSNorm(dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth - 1):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                SGLU(dim, mlp_dim, dim)
            ]))

        self.final_attn = LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout) 
        self.final_ffn = SGLU(dim, mlp_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.final_attn(x) + x

        for attn, ff in self.layers:
            for _ in range(2):
                x = self.norm(x)
                x = attn(x) + x

                x = self.norm(x)
                x = ff(x) + x

        x = self.norm(x)
        x = self.final_ffn(x) + x
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
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        #patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

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
