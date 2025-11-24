"""
Core transformer block with geometric attention.

Optional: only needed if you want a complete transformer block.
If integrating into existing model, just use geometric_attention() directly.
"""

import torch
import torch.nn as nn

from .attention import GeometricCoordinates, geometric_attention


class GeometricTransformerBlock(nn.Module):
    """
    Transformer block with geometric attention.

    Standard transformer architecture, but uses geometric_attention()
    instead of regular attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        geometric_radius: int = 2,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.geometric_radius = geometric_radius

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, D]
        coords: GeometricCoordinates,  # 2D coordinates
    ) -> torch.Tensor:
        """Forward pass with geometric attention."""
        # Self-attention
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Geometric attention
        attn_out = geometric_attention(q, k, v, coords, self.geometric_radius)

        attn_out = attn_out.reshape(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout_attn(attn_out)
        x = residual + attn_out

        # Feed-forward
        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        ff_out = self.dropout_ff(ff_out)
        x = residual + ff_out

        return x
