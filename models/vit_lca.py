# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import xformers.ops as xops

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

def select_top_tokens_consecutive(input_tensor, scores):
    B, N, _ = input_tensor.shape

    # Adjust for odd N by only considering up to the last pair
    effective_N = N - (N % 2)
    scores = scores[:, :effective_N]
    input_tensor = input_tensor[:, :effective_N, :]

    # Reshape scores to (B, N//2, 2) to consider consecutive pairs
    scores_reshaped = scores.view(B, -1, 2)

    #print(f"input_tensor = {input_tensor}")
    #print(f"scores = {scores}")
    #print(f"scores_reshaped = {scores_reshaped}")

    # Find indices of maximum scores in each consecutive pair
    _, max_indices = torch.max(scores_reshaped, dim=-1)

    #print(f"max_indices = {max_indices}")

    # Calculate global indices in the flattened version of the input tensor
    row_indices = torch.arange(B, device=scores.device)[:, None]
    global_indices = max_indices + torch.arange(0, effective_N, 2, device=scores.device)[None, :]

    #print(f"global_indices = {global_indices}")

    # Select tokens based on calculated indices
    selected_tokens = input_tensor[row_indices, global_indices]

    #print(f"selected_tokens = {selected_tokens}")

    return selected_tokens

class DownsampleMHA(nn.Module):
    def __init__(self, d_in, d_out, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(d_in, inner_dim, bias = False)
        self.to_k = nn.Linear(d_in, inner_dim, bias = False)
        self.to_v = nn.Linear(d_in, inner_dim, bias = False)

        self.score = nn.Linear(inner_dim, 1)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None, #xops.LowerTriangularMask(),
            p=self.dropout,
            scale=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)

        # Calculate a score for each (b, n) token
        score = self.score(out).squeeze(-1)

        out = self.to_out(out)

        out = out + x # Residual connection

        selected = select_top_tokens_consecutive(out, score)

        return out, selected

class CausalMHA(nn.Module):
    def __init__(self, d_in, d_out, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(d_in, inner_dim, bias = False)
        self.to_k = nn.Linear(d_in, inner_dim, bias = False)
        self.to_v = nn.Linear(d_in, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None, #xops.LowerTriangularMask(),
            p=self.dropout,
            scale=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)
        return self.to_out(out)

def _materialize_causal_mask(q, kv) -> torch.Tensor:
    dtype = q.dtype
    B, QN, H, _ = q.shape
    _, KVN, _, _ = kv.shape
    device = q.device

    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    tensor = torch.full(  # type: ignore
        torch.Size([B, H, QN, KVN]),
        dtype=create_as,
        fill_value=1,
        device=device,
    )

    mask = torch.triu(tensor, diagonal=-2).to(dtype)  # type: ignore
    mask = torch.log(mask)

    return mask.to(dtype)

class UpsampleMHA(nn.Module):
    def __init__(self, d_in_full, d_in_downsampled, d_out, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(d_in_full, inner_dim, bias = False)
        self.to_k = nn.Linear(d_in_downsampled, inner_dim, bias = False)
        self.to_v = nn.Linear(d_in_downsampled, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, unet_full, unet_downsampled):
        q = self.to_q(unet_full)
        k = self.to_k(unet_downsampled)
        v = self.to_v(unet_downsampled)

        # Repeat 2x downsampled tokens in pairs to line up with old token sequence
        k = k.repeat(1, 2, 1)
        v = v.repeat(1, 2, 1)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        attn_bias = _materialize_causal_mask(q, k)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None, #attn_bias,
            p=self.dropout,
            scale=None)

        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.down_count = 2

        # FIXME: Reduce/increase dimensions here

        self.down_layers = nn.ModuleList([])
        for _ in range(self.down_count):
            self.down_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                DownsampleMHA(d_in=dim, d_out=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

        if False:
            self.reasoning_layers = nn.ModuleList([])
            for _ in range(depth):
                self.reasoning_layers.append(nn.ModuleList([
                    nn.LayerNorm(dim),
                    CausalMHA(d_in=dim, d_out=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ]))

        self.up_layers = nn.ModuleList([])
        for _ in range(self.down_count):
            self.up_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                UpsampleMHA(d_in_downsampled=dim, d_in_full=dim, d_out=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x):
        full_list = []

        for norm, attn_and_residual, ff in self.down_layers:
            x = norm(x)
            full, x = attn_and_residual(x)

            full_list.append(full)

            x = norm(x)
            x = x + ff(x)

        if False:
            for norm, attn, ff in self.reasoning_layers:
                x = norm(x)
                x = x + attn(x)

                x = norm(x)
                x = x + ff(x)

        dx = x

        for norm, attn, ff in self.up_layers:
            x = full_list.pop()

            x = norm(x)
            x = x + attn(x, dx)

            x = norm(x)
            x = x + ff(x)

            dx = x

        return dx

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
