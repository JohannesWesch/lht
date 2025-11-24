"""
Core LHT encoder architecture.

This module defines LHTEncoder with:
- Stack of transformer layers
- Token embeddings and MLM head
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention


class LHTTransformerBlock(nn.Module):
    """
    Single Transformer block for LHT.

    Uses xFormers memory_efficient_attention.
    Input/Output: [B, N, D]
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # QKV projections and output projection
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, D]
        attention_mask: Optional[torch.Tensor] = None,  # [B, N], currently unused
    ) -> torch.Tensor:
        # ---- Self-attention ----
        residual = x
        x_norm = self.norm1(x)

        # Project to Q, K, V
        qkv = self.qkv(x_norm)  # [B, N, 3*D]
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, N, H, Dh]

        # xFormers expects [B, N, H, Dh]
        attn_out = memory_efficient_attention(q, k, v)  # [B, N, H, Dh]

        # Merge heads back
        attn_out = attn_out.reshape(B, N, self.d_model)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout_attn(attn_out)

        x = residual + attn_out

        # ---- Feed-forward ----
        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        ff_out = self.dropout_ff(ff_out)
        x = residual + ff_out

        return x


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

        # Embeddings
        self.token_embed = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.pos_embed = None  # you may use RoPE instead of absolute embeddings

        # Stack of transformer layers
        self.layers = nn.ModuleList(
            [
                LHTTransformerBlock(
                    model_cfg.d_model,
                    model_cfg.n_heads,
                    model_cfg.d_ff,
                    model_cfg.dropout,
                )
                for _ in range(model_cfg.num_layers)
            ]
        )

        # MLM head with weight tying
        self.mlm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        self.mlm_head.weight = self.token_embed.weight  # tie weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LHT encoder.

        Returns:
            dict with:
                - "hidden": final hidden states [B, N, D]
                - "mlm_logits": logits for MLM task [B, N, vocab_size]
        """
        B, N = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        x = self.token_embed(input_ids)  # [B, N, D]

        # Process through all layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # MLM head projection
        mlm_logits = self.mlm_head(x)  # [B, N, vocab_size]

        return {
            "hidden": x,
            "mlm_logits": mlm_logits,
        }
