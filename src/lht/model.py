"""
Core LHT encoder architecture.

This module defines LHTEncoder with:
- Stack of MLSWATransformerBlock layers with Multi-Level Sliding Window Attention
- Token embeddings and MLM head
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from lht.core.attention import HierarchicalPositions
from lht.core.model import MLSWATransformerBlock


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

        # We enforce ML-SWA (Multi-Level Sliding Window Attention) now.
        if getattr(model_cfg, "mlswa", None) is None:
            raise ValueError("LHTEncoder now requires an 'mlswa' config block.")

        # Embeddings with proper initialization (BERT-style)
        self.token_embed = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        self.pos_embed = None  # you may use RoPE instead of absolute embeddings

        # Stack of transformer layers
        layers = []
        for i in range(model_cfg.num_layers):
            # Use Multi-Level Sliding Window Attention (ML-SWA) for ALL layers
            layer_max_level = (
                model_cfg.mlswa.layer_max_level[i]
                if i < len(model_cfg.mlswa.layer_max_level)
                else None
            )

            layers.append(
                MLSWATransformerBlock(
                    d_model=model_cfg.d_model,
                    n_heads=model_cfg.n_heads,
                    d_ff=model_cfg.d_ff,
                    dropout=model_cfg.dropout,
                    layer_idx=i,
                    layer_max_level=layer_max_level,
                    window_size_per_level=model_cfg.mlswa.window_size_per_level,
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
        positions: Optional[HierarchicalPositions] = None,
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
        # Cascading per-level attention uses BlockMask from FlexAttention.
        # create_hierarchical_mask in attention.py creates mask based on positions.
        # If padded tokens have valid positions (e.g. dummy positions), they might participate.
        # Ideally, we should integrate attention_mask into hierarchical mask creation or pass it.

        if positions is None:
            raise ValueError(
                "LHTEncoder requires 'positions' input for hierarchical attention."
            )

        x = self.token_embed(input_ids)  # [B, N, D]

        # Process through all layers
        for layer in self.layers:
            x = layer(x, positions=positions)

        # Apply final layer norm before MLM head
        x = self.final_norm(x)

        # MLM head projection
        mlm_logits = self.mlm_head(x)

        return {
            "hidden": x,
            "mlm_logits": mlm_logits,
        }
