from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SHCNLayer(nn.Module):
    """Paper-faithful structural diffusion: (A_hat + delta * Norm(QK^T/sqrt(d)))VW."""

    def __init__(self, dim: int, heads: int, dropout: float, graph_mix_coeff: float, residual: bool) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by shcn_heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.graph_mix_coeff = graph_mix_coeff
        self.residual = residual
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        relation = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        pair_mask = mask[:, None, :, None] * mask[:, None, None, :]
        relation = relation * pair_mask
        relation = F.normalize(relation, p=2, dim=-1, eps=1e-12)
        structure = adjacency[:, None, :, :] + self.graph_mix_coeff * relation
        propagated = torch.matmul(structure, v)
        propagated = propagated.transpose(1, 2).contiguous().view(batch, length, self.dim)
        propagated = F.elu(self.out_proj(propagated))
        propagated = self.dropout(propagated)
        if self.residual:
            propagated = propagated + x
        return propagated * mask.unsqueeze(-1)


class StructuralHypergraphConvolutionNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int = 4,
        heads: int = 1,
        dropout: float = 0.1,
        graph_mix_coeff: float = 0.1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SHCNLayer(dim, heads, dropout, graph_mix_coeff, residual) for _ in range(layers)]
        )

    def forward(self, item_embeddings: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = item_embeddings * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adjacency, mask)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denominator
        return F.normalize(pooled, p=2, dim=-1, eps=1e-12)
