# https://en.wikipedia.org/wiki/Soliton_distribution

# This is a sparse matrix with random structure based on the LT code Soliton distribution

from math import sqrt
from torch import nn

class SolitonFeedForward(nn.Module):
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
        return self.net(x)
