# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

# Modified to use DynamicSlotsSoftMoE from https://github.com/lucidrains/soft-moe-pytorch
# which is also modified to use SNLinear from Apple's Sigma Reparam paper,
# which prevents vanishing gradients when training SoftMoE using FP16 and
# eliminates normalization from SoftMoE.

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

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = - 1)

def pad_to_multiple(
    tensor,
    multiple,
    dim = -1,
    value = 0
):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    #if m.is_integer(): - Not supported by Torch dynamo
    if m == math.floor(m):
        return False, tensor

    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# expert

def MyFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        SNLinear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        SNLinear(dim_hidden, dim)
    )

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def GLUFeedForward(
    dim,
    mult = 4,
    dropout = 0.
):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        SNLinear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        SNLinear(dim_hidden, dim)
    )

# main class

class DynamicSlotsSoftMoE(Module):
    def __init__(
        self,
        dim,
        *,
        num_experts = 4,
        expert_mult = 4,
        dropout = 0.,
        geglu = False
    ):
        super().__init__()
        self.num_experts = num_experts

        self.to_slot_embeds = nn.Sequential(
            SNLinear(dim, dim * num_experts, bias = False),
            Rearrange('b n (e d) -> b e n d', e = num_experts),
        )

        if geglu:
            self.experts = nn.ModuleList([
                GLUFeedForward(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                MyFeedForward(dim = dim, mult = expert_mult, dropout = dropout) for _ in range(num_experts)
            ])

    def forward(self, x, mask = None):
        """
        einstein notation
        b - batch
        n - sequence length
        e - number of experts
        s - number of slots per expert
        d - feature dimension
        """

        seq_len, is_image, num_experts = x.shape[-2], x.ndim == 4, self.num_experts

        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')

        # following Algorithm 1, with the normalization they proposed, but with scaling of both (the now popular rmsnorm + gamma)

        # dynamic slot embeds
        # first average consecutive tokens, by number of experts
        # then, for each position, project out to that number of expert slot tokens
        # there should be # slots ~= sequence length, like in a usual MoE with 1 expert

        is_padded, x = pad_to_multiple(x, num_experts, dim = -2)

        if is_padded:
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool)

            _, mask = pad_to_multiple(mask, num_experts, dim = -1, value = False)

        x_segmented = rearrange(x, 'b (n e) d -> b n e d', e = num_experts)

        if exists(mask):
            segmented_mask = rearrange(mask, 'b (n e) -> b n e', e = num_experts)
            x_segmented = x_segmented.masked_fill(~rearrange(segmented_mask, '... -> ... 1'), 0.)

        # perform a masked mean

        if exists(mask):
            num = reduce(x_segmented, 'b n e d -> b n d', 'sum')
            den = reduce(segmented_mask.half(), 'b n e -> b n 1', 'sum').clamp(min = 1e-5)
            x_consecutive_mean = num / den
            slots_mask = segmented_mask.any(dim = -1)
        else:
            x_consecutive_mean = reduce(x_segmented, 'b n e d -> b n d', 'mean')

        # project to get dynamic slots embeddings
        # could potentially inject sinusoidal positions here too before projection

        slot_embeds = self.to_slot_embeds(x_consecutive_mean)

        logits = einsum('b n d, b e s d -> b n e s', x, slot_embeds)

        # account for key padding mask

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            slots_mask = rearrange(slots_mask, 'b s -> b 1 1 s')

            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
            logits = logits.masked_fill(~slots_mask, -torch.finfo(logits.dtype).max)

        # get dispatch and combine weights (softmax across right dimensions)

        dispatch_weights = logits.softmax(dim = 1)

        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim = -1)

        # derive slots by weighted average of input tokens using the dispatch weights from above

        slots = einsum('b n d, b n e s -> e b s d', x, dispatch_weights)

        # route the slots per expert to each expert

        out = []
        for slots_per_expert, expert in zip(slots, self.experts):
            out.append(expert(slots_per_expert))

        out = torch.stack(out)

        # combine back out

        out = rearrange(out, 'e b s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)

        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')

        return out[:, :seq_len]

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

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
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

    def forward(self, x):
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

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, num_experts, expert_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                DynamicSlotsSoftMoE(dim, num_experts=num_experts, expert_mult=expert_mult, geglu=False, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
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

class SoftMoEViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, num_experts, expert_mult, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
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

        self.transformer = Transformer(dim, depth, heads, dim_head, num_experts, expert_mult, dropout)

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
