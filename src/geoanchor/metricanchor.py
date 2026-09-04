"""Lightweight frozen-feature MetricAnchor components."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ResidualMetricAdapter(nn.Module):
    def __init__(self, dim: int = 384, hidden: int = 128):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, hidden, 1)
        self.mix = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, dim, 1)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # [B,256,384] or [B,16,16,384]
        original = tokens.shape
        if tokens.ndim == 3: x = tokens.reshape(-1, 16, 16, original[-1])
        else: x = tokens
        base = x
        y = x.permute(0,3,1,2)
        y = self.out_proj(F.gelu(self.mix(F.gelu(self.in_proj(y))))).permute(0,2,3,1)
        return F.normalize((base+y).reshape(original), dim=-1)
    @property
    def parameter_count(self) -> int: return sum(p.numel() for p in self.parameters())
