# https://github.com/sap-ient-ai/FFF/blob/main/experiments/2023-12-03--fff-flat-structure.ipynb

# This is Bojan's fff-flat-experiment wrapped in a FeedForward layer

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor, log2

class FFF_v2(nn.Module):
    def __init__(self, nIn: int, nOut: int):
        super().__init__()
        self.input_width = nIn
        self.output_width = nOut
        self.depth = int(floor(log2(nIn)))  # depth is the number of decision boundaries
        nNodes = 2 ** self.depth - 1

        def create_random_unit_vectors_of(length):
            weights = torch.randn(nNodes, length)  # Initialize weights randomly
            weights = F.normalize(weights, p=2, dim=-1)  # L2-Normalize along the last dimension
            return nn.Parameter(weights)

        self.PathSelector = nn.Linear(nIn, self.depth, bias=False)
        self.Y = create_random_unit_vectors_of(length=nOut)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        if x.dim() == 3:
            # Reshape x from [nBatch, seq_len, nIn] to [nBatch * seq_len, nIn]
            x = x.view(-1, self.input_width)

        λ = self.PathSelector(x)
        branch_choice = (λ > 0).long()

        nBatch = λ.shape[0]

        indices = torch.empty((nBatch, self.depth), dtype=torch.long, device=x.device)
        current_node = torch.zeros(nBatch, dtype=torch.long, device=x.device)
        for i in range(self.depth):
            indices[:, i] = current_node

            current_node = (current_node * 2) + 1 + branch_choice[:, i]

        y = torch.einsum("b i, b i j -> b j", λ, self.Y[indices])

        if original_shape[1] != self.input_width:  # Check if input was 3D
            # Reshape y back to the original 3D shape [nBatch, seq_len, nOut]
            y = y.view(original_shape[0], original_shape[1], self.output_width)

        return y


# I added this part
class Bojan_Flat_FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            FFF_v2(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            FFF_v2(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
