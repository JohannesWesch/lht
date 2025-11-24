"""
Core LHT encoder architecture.

This module defines LHTEncoder with:
- Stack of GeometricTransformerBlock layers
- Token embeddings and MLM head
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from lht.core.attention import GeometricCoordinates
from lht.core.model import GeometricTransformerBlock


class LHTEncoder(nn.Module):
    """
    Learned Hierarchical Transformer encoder.

    Input: token ids [B, N]
    Output: contextual representations [B, N, D]
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.cfg = config

        # Access model config properly
        model_cfg = config.model if hasattr(config, "model") else config

        # We enforce geometric attention now.
        if getattr(model_cfg, "geometry", None) is None:
            raise ValueError("LHTEncoder now requires a 'geometry' config block.")

        # Embeddings with proper initialization (BERT-style)
        self.token_embed = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.pos_embed = None  # you may use RoPE instead of absolute embeddings

        # Stack of transformer layers
        layers = []
        for i in range(model_cfg.num_layers):
            # Use Geometric Attention for ALL layers
            layer_max_level = (
                model_cfg.geometry.layer_max_level[i]
                if i < len(model_cfg.geometry.layer_max_level)
                else None
            )
            layer_radius = model_cfg.geometry.manhattan_radius

            layers.append(
                GeometricTransformerBlock(
                    d_model=model_cfg.d_model,
                    n_heads=model_cfg.n_heads,
                    d_ff=model_cfg.d_ff,
                    dropout=model_cfg.dropout,
                    manhattan_radius=layer_radius,
                    layer_idx=i,
                    layer_max_level=layer_max_level,
                    window_size_per_level=model_cfg.geometry.window_size_per_level,
                )
            )

        self.layers = nn.ModuleList(layers)

        # Final layer norm (critical for stable logits!)
        self.final_norm = nn.LayerNorm(model_cfg.d_model)

        # MLM head with proper initialization (no weight tying for stability)
        self.mlm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        nn.init.normal_(self.mlm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        coords: Optional[GeometricCoordinates] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LHT encoder.

        Returns:
            dict with:
                - "hidden": final hidden states [B, N, D]
                - "mlm_logits": logits for MLM task [B, N, vocab_size]
        """
        B, N = input_ids.shape

        # Note: attention_mask is traditional padding mask.
        # Geometric block uses BlockMask from FlexAttention which handles causal/sparse logic.
        # But we still need to handle padding tokens so they don't attend or get attended to?
        # create_geometric_mask in attention.py doesn't explicitly handle padding mask passed here.
        # FlexAttention usually handles padding if BlockMask says so or via separate key_padding_mask (if supported).
        # Currently, geometric_attention.py creates a mask based on coords.
        # If padded tokens have valid coords (e.g. dummy coords), they might participate.
        # Ideally, we should integrate attention_mask into geometric mask creation or pass it.

        if coords is None:
            raise ValueError(
                "LHTEncoder requires 'coords' input for geometric attention."
            )

        x = self.token_embed(input_ids)  # [B, N, D]

        # Process through all layers
        for layer in self.layers:
            x = layer(x, coords=coords)

        # Apply final layer norm before MLM head
        x = self.final_norm(x)

        # MLM head projection
        mlm_logits = self.mlm_head(x)

        return {
            "hidden": x,
            "mlm_logits": mlm_logits,
        }
