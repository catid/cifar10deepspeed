# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .srmsnorm import FastSimpleRMSNorm

import math

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )

use_qkvpacked = True

if use_qkvpacked:

    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func

    class LSA(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super().__init__()
            self.dropout = dropout
            self.heads = heads
            inner_dim = dim_head * self.heads

            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )

            alibi_slopes = torch.tensor(get_alibi_slopes(heads))
            self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        def forward(self, x):
            qkv = self.to_qkv(x)
            qkv = rearrange(qkv, 'b n (a h d) -> b n a h d', a = 3, h = self.heads)  

            #alibi_slopes = self.alibi_slopes.to(x.device)
            alibi_slopes = None
            # Note: Alibi slopes are only available in a broken newer version of the package
            # Issue is here: https://github.com/Dao-AILab/flash-attention/issues/867
            out = flash_attn_qkvpacked_func(qkv, dropout_p=self.dropout, softmax_scale=None, causal=False, window_size=(-1, -1))

            out = rearrange(out, 'b n h d -> b n (h d)', h = self.heads)
            return self.to_out(out)

else:

    # FIXME: This is broken

    from .flash_attn_triton import FlashAttnFunc

    class LSA(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
            super().__init__()
            self.dropout = dropout
            self.heads_q = heads
            self.heads_kv = heads
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

            q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads_q)
            k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads_kv)
            v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads_kv)

            _, QN, H, _ = q.shape
            _, KN, _, _ = k.shape

            alibi_slopes = torch.tensor(get_alibi_slopes(H), device=x.device)
            alibi_slopes = alibi_slopes.view(1, -1, 1, 1)
            range_k = torch.arange(KN, dtype=alibi_slopes.dtype, device=x.device)
            range_k = range_k.view(1, 1, 1, -1).expand(1, H, 1, -1)
            range_q = torch.arange(QN, dtype=alibi_slopes.dtype, device=x.device)
            range_q = range_q.view(1, 1, 1, -1).expand(1, H, 1, -1)
            alibi_slopes = alibi_slopes * range_tensor

            out = FlashAttnFunc.apply(q, k, v, alibi_slopes)

            out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads_q)
            return self.to_out(out)

class SGLU(nn.Module):
    def __init__(self, d_in=64, mlp_dim=256, d_out=64, bias=False):
        super().__init__()

        self.in_u = nn.Linear(d_in, mlp_dim, bias=bias)
        self.in_v = nn.Linear(d_in, mlp_dim, bias=bias)
        self.out_proj = nn.Linear(mlp_dim, d_out, bias=bias)

    def forward(self, x):
        return self.out_proj(self.in_u(x) * self.in_v(x))

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = FastSimpleRMSNorm(dim)
        for d in range(depth - 1):
            #if d == 1:
            #    from product_key_memory import PKM
            #    ff = PKM(dim, heads = heads, dim_head = 128, num_keys = 256, topk = 32)
            #else:
            #    ff = FeedForward(dim, mlp_dim, dropout = dropout)
            self.layers.append(nn.ModuleList([
                LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                SGLU(d_in=dim, mlp_dim=mlp_dim, d_out=dim)
            ]))

        # We split this one across first/last
        self.final_attn = LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout) 
        self.final_ffn = SGLU(d_in=dim, mlp_dim=mlp_dim, d_out=dim)

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
