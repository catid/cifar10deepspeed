# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from fla.ops import fused_recurrent_gla
from .srmsnorm import FastSimpleRMSNorm

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        y = self.ffn(x)
        return y

class Hgrn2(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=64,
        expand_ratio=2,
    ):
        super().__init__()

        self.expand_ratio = expand_ratio
        forget_dim = num_heads * expand_ratio
        self.num_heads = num_heads

        self.q_proj = nn.Linear(embed_dim, forget_dim, bias=False)
        self.f_proj = nn.Linear(embed_dim, forget_dim, bias=False)
        #self.i_proj = nn.Linear(embed_dim, embed_dim, bias=False) This is what HGRN2 paper does
        self.i_proj = nn.Linear(embed_dim, forget_dim, bias=False)

        #self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False) This is what HGRN2 paper does
        self.o_proj = nn.Linear(forget_dim, embed_dim, bias=False)
        self.in_act = nn.GELU()

        # Added an output activation, which seems to not hurt accuracy scores at all and is
        # not in the original HGRN2 paper but is suggested by this paper:
        # "The Illusion of State in State-Space Models" https://arxiv.org/abs/2404.08819
        # It may help for language modeling, but I'm not sure.
        self.out_act = nn.Tanh()

    def forward(self, x, lower_bound=0.0):
        q = self.in_act(self.q_proj(x))

        f = self.f_proj(x)
        g = lower_bound + (1 - lower_bound) * f.sigmoid()

        i = self.i_proj(x)

        k = 1 - g

        g = torch.log(g + 1e-6) # Add epsilon to avoid log(0)

        q, k, i, g = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, i, g))

        # We do not have a recurrent state here because it is just one sequence of tokens
        o, _ = fused_recurrent_gla(q, k, i, g, initial_state=None, output_final_state=False)

        o = rearrange(o, 'b h l d -> b l (h d)')
        o = self.o_proj(self.out_act(o))

        return o

class Hgrn2Former(nn.Module):
    def __init__(self, dim, depth, num_heads, expand_ratio, mlp_dim, dropout = 0.):
        super().__init__()

        self.norm = FastSimpleRMSNorm(dim)

        lb_dim = num_heads * expand_ratio

        self.lower_bounds = nn.Parameter(
            torch.ones(depth, lb_dim), requires_grad=True
        )

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Hgrn2(embed_dim=dim, num_heads=num_heads, expand_ratio=expand_ratio),
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
        return self.norm(x)

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

class Hgrn2ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, num_heads, expand_ratio, mlp_dim, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.):
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

        self.transformer = Hgrn2Former(dim, depth, num_heads, expand_ratio, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        #x = x[:, :-1, :]

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((x, cls_tokens), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, -1, :]

        x = self.to_latent(x)
        return self.mlp_head(x)
