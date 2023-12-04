# https://api.mdsoar.org/server/api/core/bitstreams/360d95b8-2480-4440-92bd-c69fdf2a92db/content

# "Cyclic Sparsely Connected Architectures for Compact Deep Convolutional Neural Networks"

from math import sqrt
from torch import nn

class CsciLinear(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()

    def forward(self, x):
        return self.net(x)

class CsciFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.CsciLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.CsciLinear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
