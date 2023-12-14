# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_for_small_dataset.py

from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# BaseConv

from typing import List, Union

class ShortConvolution(nn.Module):
    """
    Simple wrapper around nn.Conv1d that accepts dimension last. 
    """

    def __init__(
        self, 
        d_model: int,
        kernel_size: int
    ): 
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        l = x.size(1)
        y = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)
        return y 

def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)

class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed 
    filter of length l_max.
    The filter is learned during training and is applied using FFT convolution.
    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
    Returns:
        y: (b, l, d) tensor
    """
    def __init__(
        self,
        d_model: int,
        l_max: int,
        **kwargs,
    ):
        """
        Initializes the LongConvolution module.
        Args:
            d_model (int): The number of expected features in the input and output.
            l_max (int): The maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model 
        self.filter = nn.Parameter(torch.randn(self.d_model, l_max), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]

class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
        d_emb (int, optional): The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine). Defaults to 3.
        d_hidden (int, optional): The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (PositionalEmbedding): The positional embedding layer.
        mlp (nn.Sequential): The MLP that parameterizes the implicit filter.

    """

    
    def __init__(
        self,
        d_model: int,
        l_max: int,
        d_emb: int=3, 
        d_hidden: int = 16,
        **kwargs,
    ):
        """
        Long convolution with implicit filter parameterized by an MLP.

        
        """
        super().__init__()
        self.d_model = d_model 
        self.d_emb = d_emb 


        assert (
            d_emb % 2 != 0 and d_emb >= 3
        ), "d_emb must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(d_emb, l_max)

        # final linear layer
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )


    def filter(self, l: int, *args, **kwargs):
        k = self.mlp(self.pos_emb(l))

        return k.transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)

class BaseConv(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        kernel_size: Union[int, List[int]]=3,
        layer_idx: int=None,
        implicit_long_conv: bool=True,
        use_act=False,
        **kwargs
    ):
        super().__init__()
      
        self.use_act = use_act
        self.d_model = d_model
        self.l_max = l_max
        self.layer_idx=layer_idx

        self.projection = nn.Linear(self.d_model,  self.d_model)
        if self.use_act:
            self.act = nn.SiLU() 
        
        # support for different kernel sizes per layer
        if isinstance(kernel_size, List):
            if layer_idx is  None or layer_idx >= len(kernel_size):
                raise ValueError("kernel_size must be an int or a list of ints with length equal to the number of layers")
            kernel_size = kernel_size[layer_idx]

        # prepare convolution
        if kernel_size == -1:
            conv = ImplicitLongConvolution if implicit_long_conv else LongConvolution
            self.conv = conv(d_model, l_max=l_max)
        else:
            self.conv = ShortConvolution(d_model, kernel_size=kernel_size)

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        u_conv = self.conv(u)
        u_proj = self.projection(u)
        if self.use_act:
            y = self.act(u_conv) * self.act(u_proj)
        else:
            y = u_conv * u_proj
        return y + u

# Based

import opt_einsum as oe
import math

def init_feature_map(feature_map: str='none', **kwargs: any):
    """
    Initialize query and key mapping for linear attention
    """
    if feature_map in [None, 'none', 'identity']:
        return FeatureMap(**kwargs)
    # Taylor series approximations to exp(x)
    elif feature_map == 'taylor_exp':
        return TaylorExp(**kwargs) 
    else:
        raise NotImplementedError(f'Sorry "{feature_map}" feature map not implemented.')
        
        
class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """
    def __init__(self, 
                 input_dim: int,                 
                 temp: int = None,
                 head_dim_idx: int = -1, 
                 eps: float = 1e-12, 
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx     
        self.temp = 1. if temp is None else temp
        self.eps = eps
        
    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x

    
class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """
    def __init__(self, input_dim: int, **kwargs: any):
        super().__init__(input_dim, **kwargs)
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(self.input_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(self.input_dim, self.input_dim, -1)
        
    # Running these in parallel
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(start_dim=-2) / self.r2
        return torch.cat([torch.ones(x[..., :1].shape).to(x.device), 
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)
        
    def forward_mem_save(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x) s.t. f(x)^T f(x') = 1 + x^Tx' + (x^Tx')^2 / 2
        -> Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        # Slow but memory-saving way to compute 2nd-order terms; how do w/o outer-product first?
        x2  = oe.contract('...m,...n->...mn', x, x) / self.input_dim
        x2d = torch.diagonal(x2, dim1=-2, dim2=-1) / self.r2
        x2  = x2[..., self.tril_indices[0], self.tril_indices[1]]
        x   = torch.cat([torch.ones(x[..., :1].shape).to(x.device), 
                         x / self.rd, x2d, x2], dim=-1)
        return x 


class Based(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 12,
        num_heads: int = 12,
        feature_name: "str" = "taylor_exp",
        eps: float = 1e-12,
        causal: bool = True,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max

        # linear attention 
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_key_value_heads
        self.causal=causal
        feature_map_kwargs = {
            'input_dim': self.feature_dim,
            'head_dim_idx': -1,
            'temp': 1.,
            'eps': 1e-12
        }
        self.feature_map = init_feature_map(feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps


    def forward(self, hidden_states: torch.Tensor, filters: torch.Tensor=None, *args, **kwargs):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)

        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
        
        # Compute attention
        if self.causal:
            y = ((q * (k * v).cumsum(dim=2)).sum(dim=-1) / 
                 ((q * k.cumsum(dim=2)).sum(dim=-1) + self.eps))
        else:
            y = ((q * (k * v).sum(dim=2, keepdim=True)).sum(dim=-1) /
                 ((q * k.sum(dim=2, keepdim=True)).sum(dim=-1) + self.eps))
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.proj_o(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype) 




# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Block(nn.Module):
    def __init__(self, index, l_max, d_model, kernel_size, feature_dim, num_key_value_heads, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.base_conv = BaseConv(d_model=d_model, l_max=l_max, kernel_size=kernel_size, layer_idx=index, implicit_long_conv=True)
        self.based = Based(d_model=d_model, l_max=l_max, feature_dim=feature_dim, num_key_value_heads=num_key_value_heads, num_heads=num_heads, feature_name="taylor_exp", causal=True)

    def forward(self, x, **kwargs):
        # Block norm
        y = self.norm(x)

        y = self.base_conv(y)
        y = self.based(y)

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

class ViBased(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, d_model, kernel_size, feature_dim, num_key_value_heads, num_heads, n_layers, pool = 'cls', channels = 3, emb_dropout = 0.):
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

        l_max = num_classes + 1

        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(Block(i, l_max, d_model, kernel_size=kernel_size, feature_dim=feature_dim, num_key_value_heads=num_key_value_heads, num_heads=num_heads))

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
