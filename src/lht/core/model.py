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

    Supports per-layer hierarchy level control via max_level parameter.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        manhattan_radius: int = 1,
        layer_idx: int = None,
        layer_max_level: int = None,
        window_size_per_level: list = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.manhattan_radius = manhattan_radius  # geometric parent-child (usually 1)
        self.layer_idx = layer_idx
        self.layer_max_level = layer_max_level
        self.window_size_per_level = window_size_per_level  # [512, 64, 16]

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
        max_level: int = None,  # Override layer_max_level if provided
        window_size_per_level: list = None,  # Override self.window_size_per_level if provided
    ) -> torch.Tensor:
        """Forward pass with geometric attention.

        Args:
            x: Input tensor [B, N, D]
            coords: Geometric coordinates for all tokens
            max_level: Max hierarchy level for this layer (overrides self.layer_max_level)
            window_size_per_level: Window sizes [512, 64, 16] (overrides self.window_size_per_level)
        """
        # Self-attention
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Determine max_level: explicit arg > layer_max_level > None (all levels)
        effective_max_level = (
            max_level if max_level is not None else self.layer_max_level
        )

        # Determine window sizes: explicit arg > self.window_size_per_level > None
        effective_windows = (
            window_size_per_level
            if window_size_per_level is not None
            else self.window_size_per_level
        )

        # Geometric attention with per-layer level control
        attn_out = geometric_attention(
            q,
            k,
            v,
            coords,
            radius=self.manhattan_radius,
            max_level=effective_max_level,
            window_size_per_level=effective_windows,
        )

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
