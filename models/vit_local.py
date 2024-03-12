# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

# This is modified from the original implementation to add a local window size parameter.
# The defaults use local attention with a window size of 8 instead of 65 (baseline).

# Then it does another local attention with full columns that are selected based on the
# token scores from the previous layer.  Token scores are the sum of the attention scores
# for all heads.  By default we have a tile size of 8 so we attend only 1/8 of the tokens
# in the second pass.  The same weights are used for both passes.

# The idea is that we use two passes:
# (1) The local attention identifies the most interesting tokens in each tile.
#     This is standard local attention with a window size of 8.
# (2) Then we attend to the most interesting tokens in the second pass, which were
#     discovered in the first pass.  This part is the new idea.

# The speed of this implementation is not great, but an optimized version could be
# implemented in the future.  Note that we only attend 12% of the tokens in the first
# pass, and only 12% of the tokens in the second pass.  They might be the same tokens
# as the first round but that is not necessarily a bad thing.

# We can also window the attention of the second pass to make it scale linearly with
# the number of tokens, turning it into a linear attention algorithm.  For CIFAR-10
# this window is set to 32, but we might want to pick other values.

# My result from this experiment is that we get +0.2% accuracy over baseline attending
# somewhere between only 12-25% of the baseline tokens, and not using more parameters.


import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class SpectralNormedWeight(nn.Module):
    """SpectralNorm Layer. First sigma uses SVD, then power iteration."""

    def __init__(
        self,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        with torch.no_grad():
            _, s, vh = torch.linalg.svd(self.weight, full_matrices=False)

        self.register_buffer("u", vh[0])
        self.register_buffer("spectral_norm", s[0] * torch.ones(1))

    def get_sigma(self, u: torch.Tensor, weight: torch.Tensor):
        with torch.no_grad():
            v = weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            if self.training:
                self.u.data.copy_(u)

        return torch.einsum("c,cd,d->", v, weight, u)

    def forward(self):
        """Normalize by largest singular value and rescale by learnable."""
        sigma = self.get_sigma(u=self.u, weight=self.weight)
        if self.training:
            self.spectral_norm.data.copy_(sigma)

        return self.weight / sigma

class SNLinear(nn.Linear):
    """Spectral Norm linear from sigmaReparam.

    Optionally, if 'stats_only' is `True`,then we
    only compute the spectral norm for tracking
    purposes, but do not use it in the forward pass.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.stats_only = stats_only
        self.init_multiplier = init_multiplier

        self.init_std = 0.02 * init_multiplier
        nn.init.trunc_normal_(self.weight, std=self.init_std)

        # Handle normalization and add a learnable scalar.
        self.spectral_normed_weight = SpectralNormedWeight(self.weight)
        sn_init = self.spectral_normed_weight.spectral_norm

        # Would have set sigma to None if `stats_only` but jit really disliked this
        self.sigma = (
            torch.ones_like(sn_init)
            if self.stats_only
            else nn.Parameter(
                torch.zeros_like(sn_init).copy_(sn_init), requires_grad=True
            )
        )

        self.register_buffer("effective_spectral_norm", sn_init)
        self.update_effective_spec_norm()

    def update_effective_spec_norm(self):
        """Update the buffer corresponding to the spectral norm for tracking."""
        with torch.no_grad():
            s_0 = (
                self.spectral_normed_weight.spectral_norm
                if self.stats_only
                else self.sigma
            )
            self.effective_spectral_norm.data.copy_(s_0)

    def get_weight(self):
        """Get the reparameterized or reparameterized weight matrix depending on mode
        and update the external spectral norm tracker."""
        normed_weight = self.spectral_normed_weight()
        self.update_effective_spec_norm()
        return self.weight if self.stats_only else normed_weight * self.sigma

    def forward(self, inputs: torch.Tensor):
        weight = self.get_weight()
        return F.linear(inputs, weight, self.bias)

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

def create_local_attention_mask(device, seq_length, local_window_size):
    mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
    mask = mask & torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device), diagonal=-(local_window_size - 1))
    return mask

def create_topk_attention_mask(device, seq_length, prev_token_scores, tile_size=8, local_window_size=None):
    num_tiles = seq_length // tile_size
    indices = []
    for i in range(num_tiles):
        start_idx = i * tile_size
        end_idx = (i + 1) * tile_size
        tile_scores = prev_token_scores[:, start_idx:end_idx]
        _, top_idx = tile_scores.topk(1, dim=-1)
        indices.append(start_idx + top_idx.squeeze(-1))
    indices = torch.stack(indices, dim=1)

    # Create a new mask with ones in the specified columns
    mask = torch.zeros(prev_token_scores.shape[0], seq_length, seq_length, dtype=torch.bool, device=device)
    for b in range(prev_token_scores.shape[0]):
        mask[b, :, indices[b]] = True

    # Set the diagonal to True (1)
    mask = mask | torch.eye(seq_length, dtype=torch.bool, device=device)

    # Apply the causal triangular mask
    mask = torch.tril(mask)

    if local_window_size is not None:
        mask = mask & torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device), diagonal=-(local_window_size - 1))

    return mask

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., local_window_size = 8, tile_size = 8, tile_window_size = 32):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = SNLinear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            SNLinear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.scale = dim_head ** -0.5
        self.scale_inverse = dim_head ** 0.5
        self.option = 'baseline'
        self.window_size = local_window_size
        self.tile_size = tile_size
        self.tile_window_size = tile_window_size

    def forward(self, x, prev_token_scores = None, generate_token_scores = False):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if self.option == 'baseline':
            dots = dots * self.temperature.exp()
        elif self.option == 'sqrtd':
            dots = dots / torch.clamp(dots.std(dim=-1, keepdim=True), max=self.scale_inverse, min=1e-6)
        elif self.option == 'inf':
            dots = dots / torch.clamp(dots.std(dim=-1, keepdim=True), min=1e-6)
        else:
            exit(1)

        if prev_token_scores is None:
            mask = create_local_attention_mask(dots.device, dots.shape[-1], self.window_size)
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)
        else:
            mask = create_topk_attention_mask(dots.device, dots.shape[-1], prev_token_scores, tile_size=self.tile_size, local_window_size=self.tile_window_size)
            mask = mask.unsqueeze(1)
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        if generate_token_scores:
            token_scores = attn.sum(dim=-2).sum(dim=-2)
        else:
            token_scores = None

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), token_scores

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., local_window_size = 8, tile_size = 8, tile_window_size = 32):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout, local_window_size=local_window_size, tile_size=tile_size, tile_window_size=tile_window_size),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            y, token_scores = attn(x, prev_token_scores=None, generate_token_scores=True)
            x = y + x

            y, token_scores = attn(x, prev_token_scores=token_scores, generate_token_scores=False)
            x = y + x

            x = ff(x) + x
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

class LocalViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., local_window_size = 8, tile_size = 8, tile_window_size = 32):
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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, local_window_size=local_window_size, tile_size=tile_size, tile_window_size=tile_window_size)

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
