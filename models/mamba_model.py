# This replaces the S4D kernel from S4Model with Mamba

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm import Mamba

class MambaModel(nn.Module):

    def __init__(
        self,
        d_input=3,
        d_output=10,
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=8,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                #S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.view(x.shape[0], x.shape[1], -1) # [256, 3, 1024]
        x = x.permute(0, 2, 1) # interleaved

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Adapt data to Mamba layer
            z = rearrange(z, "b d l -> b l d")
            z = layer(z)
            z = rearrange(z, "b l d -> b d l")

            # Dropout on the output of the block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
