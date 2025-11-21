"""
Core LHT encoder architecture.

This module defines LHTEncoder, which wires together:
- local token layers (sliding window),
- routers for token→sentence and sentence→section,
- mid-level sentence-aware layers,
- global hierarchical layers with HDT-style child/parent/neighbour masks.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention_masks import (
    HierarchicalBias,
    SentenceAwareBias,
    build_local_attention_bias,
)
from .routers import SentenceToSectionRouter, TokenToSentenceRouter


class LHTEncoder(nn.Module):
    """
    Learned Hierarchical Transformer encoder.

    Input: token ids [B, N]
    Output: contextual representations [B, N, D] + router diagnostics.
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.cfg = config

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = None  # you may use RoPE instead of absolute embeddings

        # Local / mid / global transformers: you'll define actual blocks later.
        self.local_layers = nn.ModuleList()
        self.mid_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        # TODO: construct Transformer blocks here.

        # Routers
        self.router_tok2sent = TokenToSentenceRouter(config.router, config.d_model)
        self.router_sent2sec = SentenceToSectionRouter(config.router, config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full LHT encoder.

        Returns a dict with:
          - "hidden": final hidden states [B, N, D]
          - router outputs / diagnostics / loss terms (to be used in training loop)
        """
        x = self.token_embed(input_ids)  # [B, N, D]

        # ------ Local layers (1–3): sliding window -------
        local_bias = build_local_attention_bias(
            seq_len=x.size(1),
            window_size=self.cfg.attention.local_window_tokens,
            device=x.device,
        )
        for layer in self.local_layers:
            x = layer(x, attn_bias=local_bias, attention_mask=attention_mask)

        # ------ Router 1: token -> sentence -------
        router1_out = self.router_tok2sent(x, attention_mask=attention_mask)
        token2sent = router1_out["token2sent"]
        is_sent_head = router1_out["is_sent_head"]

        # ------ Mid layers (4–6): sentence-aware masking -------
        sent_bias = SentenceAwareBias(
            token2sent=token2sent,
            is_sent_head=is_sent_head,
            use_doc_token=self.cfg.attention.use_doc_token,
        )
        for layer in self.mid_layers:
            x = layer(x, attn_bias=sent_bias, attention_mask=attention_mask)

        # ------ Router 2: sentence -> section -------
        router2_out = self.router_sent2sec(
            x,
            token2sent=token2sent,
            is_sent_head=is_sent_head,
        )
        token2sec = router2_out["token2sec"]
        is_sec_head = router2_out["is_sec_head"]

        # ------ Global layers (7–12): hierarchical masking -------
        hier_bias = HierarchicalBias(
            token2sent=token2sent,
            token2sec=token2sec,
            is_sent_head=is_sent_head,
            is_sec_head=is_sec_head,
            use_doc_token=self.cfg.attention.use_doc_token,
            neighbour_sentences=self.cfg.attention.neighbour_sentences,
            neighbour_sections=self.cfg.attention.neighbour_sections,
        )
        for layer in self.global_layers:
            x = layer(x, attn_bias=hier_bias, attention_mask=attention_mask)

        return {
            "hidden": x,
            "router_tok2sent": router1_out,
            "router_sent2sec": router2_out,
        }
