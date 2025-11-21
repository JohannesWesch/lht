"""
Core LHT encoder architecture with schedule-driven hierarchy.

This module defines LHTEncoder with:
- Single stack of transformer layers (no hard-coded local/mid/global)
- Schedule-driven router placement (configured via at_layer)
- Dynamic attention bias selection based on current hierarchy depth
- Support for arbitrary K-level hierarchies
"""

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention

from .attention_masks import (
    HierarchicalBias,
    SentenceAwareBias,
    build_local_attention_bias,
)
from .hierarchy import HierarchyManager

if TYPE_CHECKING:
    from xformers.ops.fmha.attn_bias import AttentionBias


class LHTTransformerBlock(nn.Module):
    """
    Single Transformer block for LHT.

    Uses xFormers memory_efficient_attention with an AttentionBias.
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
        attn_bias: Optional["AttentionBias"] = None,
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
        attn_out = memory_efficient_attention(
            q, k, v, attn_bias=attn_bias
        )  # [B, N, H, Dh]

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
    Learned Hierarchical Transformer encoder with flexible architecture.

    Unlike fixed "local → mid → global" stages, uses a single layer stack
    with schedule-driven router placement. The at_layer config determines
    when each router fires, making the architecture fully flexible.

    Input: token ids [B, N]
    Output: contextual representations [B, N, D] + hierarchy state
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

        # Single stack of transformer layers (no local/mid/global separation)
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

        # Hierarchy Manager: handles K abstraction levels
        self.hierarchy_mgr = HierarchyManager(config.hierarchy, model_cfg.d_model)
        self.num_levels = self.hierarchy_mgr.num_levels

        # Build router schedule: which layer triggers which router
        self.router_schedule = {
            lvl_cfg.at_layer: level_idx + 1
            for level_idx, lvl_cfg in enumerate(config.hierarchy.levels)
        }

        # MLM head with weight tying
        self.mlm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        self.mlm_head.weight = self.token_embed.weight  # tie weights

    def _pick_bias_for_layer(
        self,
        layer_idx: int,
        hier_state: Dict,
        seq_len: int,
        device: torch.device,
    ):
        """
        Dynamically select attention bias based on current hierarchy depth.

        Before any routers: local window attention
        After 1st router: sentence-aware attention
        After 2+ routers: full hierarchical attention

        Args:
            layer_idx: current layer index (1-indexed)
            hier_state: current hierarchy state
            seq_len: sequence length
            device: torch device

        Returns:
            AttentionBias object for this layer
        """
        level_ids = hier_state.get("level_ids", {})
        is_heads = hier_state.get("is_heads", {})
        num_levels_so_far = len(level_ids)

        if num_levels_so_far == 0:
            # Before first router → local window attention
            return build_local_attention_bias(
                seq_len=seq_len,
                window_size=self.cfg.attention.local_window_tokens,
                device=device,
            )
        elif num_levels_so_far == 1:
            # After first router only → sentence-aware attention
            return SentenceAwareBias(
                token2sent=level_ids[1],
                is_sent_head=is_heads[1],
                use_doc_token=self.cfg.attention.use_doc_token,
            )
        else:
            # After 2+ routers → full hierarchical attention
            return HierarchicalBias(
                level_ids=list(level_ids.values()),
                is_heads=list(is_heads.values()),
                use_doc_token=self.cfg.attention.use_doc_token,
                neighbour_sentences=self.cfg.attention.neighbour_sentences,
                neighbour_sections=self.cfg.attention.neighbour_sections,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Schedule-driven forward pass through LHT encoder.

        The pass is controlled entirely by config:
        - Total layers from config.model.num_layers
        - Router placement from config.hierarchy.levels[*].at_layer
        - Attention bias chosen dynamically based on hierarchy depth

        Returns:
            dict with:
                - "hidden": final hidden states [B, N, D]
                - "hierarchy": full hierarchy state
                - "router_ratio_loss": sum of all ratio losses
        """
        B, N = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        x = self.token_embed(input_ids)  # [B, N, D]

        # Initialize empty hierarchy state
        hier_state = {
            "level_ids": {},
            "is_heads": {},
            "ratio_losses": {},
        }

        # Process through all layers with schedule-driven routing
        model_cfg = self.cfg.model if hasattr(self.cfg, "model") else self.cfg
        for layer_idx in range(1, model_cfg.num_layers + 1):
            # 1) Choose attention bias for this layer
            attn_bias = self._pick_bias_for_layer(
                layer_idx=layer_idx,
                hier_state=hier_state,
                seq_len=N,
                device=device,
            )

            # 2) Run transformer block
            x = self.layers[layer_idx - 1](
                x,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
            )

            # 3) Check if router should fire after this layer
            if layer_idx in self.router_schedule:
                target_level = self.router_schedule[layer_idx]
                hier_state = self.hierarchy_mgr.update_single_level(
                    hidden=x,
                    attention_mask=attention_mask,
                    target_level=target_level,
                    state=hier_state,
                )

        # Aggregate losses
        ratio_losses = list(hier_state["ratio_losses"].values())
        router_ratio_loss = (
            torch.stack(ratio_losses).sum()
            if ratio_losses
            else torch.tensor(0.0, device=device)
        )

        # MLM head projection
        mlm_logits = self.mlm_head(x)  # [B, N, vocab_size]

        return {
            "hidden": x,
            "mlm_logits": mlm_logits,
            "hierarchy": hier_state,
            "router_ratio_loss": router_ratio_loss,
        }
