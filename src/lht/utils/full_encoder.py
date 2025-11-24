"""
Full encoder with embeddings and MLM head.

This is optional - only needed if you want a complete standalone encoder.
"""

import torch
import torch.nn as nn

from ..core import GeometricCoordinates, GeometricTransformerBlock


class GeometricLHTEncoder(nn.Module):
    """
    Complete encoder with embeddings + geometric attention + MLM head.

    Optional: use this if you want a standalone encoder.
    Otherwise, just integrate core.GeometricTransformerBlock into your own model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        geometric_radius: int = 2,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                GeometricTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    geometric_radius=geometric_radius,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.mlm_head.weight = self.token_embed.weight  # weight tying

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, N]
        coords: GeometricCoordinates,  # coordinates
    ) -> dict:
        """
        Forward pass.

        Returns:
            dict with 'hidden' and 'mlm_logits'
        """
        x = self.token_embed(input_ids)

        for layer in self.layers:
            x = layer(x, coords=coords)

        x = self.output_norm(x)
        mlm_logits = self.mlm_head(x)

        return {
            "hidden": x,
            "mlm_logits": mlm_logits,
        }
