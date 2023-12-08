# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from mamba_ssm import Mamba

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Block(nn.Module):
    def __init__(self, index, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=index)

    def forward(self, x, **kwargs):
        # Block norm
        y = self.norm(x)

        # Mamba has big MLPs at the front and back, sandwiching a state space model in the middle
        y = self.mamba(y)

        # Residual connection
        y += x

        return y

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

class ViM(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, d_model, d_state, d_conv, expand, n_layers, pool = 'cls', channels = 3, emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        #patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = d_model, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(Block(i, d_model, d_state, d_conv, expand))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        # Concatenate the scratch token to the sequence
        x = torch.cat((x, cls_tokens), dim=1)

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, -1]

        x = self.to_latent(x)
        return self.mlp_head(x)
