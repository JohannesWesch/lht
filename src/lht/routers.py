"""
Routers for learning hierarchical boundaries.

This module implements H-Net-style routing:
- TokenToSentenceRouter: predicts sentence head tokens.
- SentenceToSectionRouter: predicts section head tokens.

Both use straight-through estimation so gradients can flow through
discrete boundary decisions.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class TokenToSentenceRouter(nn.Module):
    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        # Very simple example; you can swap in a conv or more context.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        hidden: [B, N, D] from local layers.

        Returns:
          - is_sent_head: [B, N] float (STE binarised)
          - token2sent:   [B, N] long, sentence index for each token
          - aux losses (ratio loss) can be computed in training loop.
        """
        logits = self.mlp(hidden).squeeze(-1)  # [B, N]
        probs = torch.sigmoid(logits)

        # straight-through binarisation
        hard = (probs > 0.5).float()
        is_head = hard + probs - probs.detach()

        # very simple sentence ids via cumulative sum
        token2sent = torch.cumsum(is_head, dim=-1) - 1  # [-1, 0, 1, 1, 2, ...]

        if attention_mask is not None:
            token2sent = token2sent * attention_mask + (-1) * (1 - attention_mask)

        return {
            "logits": logits,
            "probs": probs,
            "is_sent_head": is_head,
            "token2sent": token2sent.long(),
        }


class SentenceToSectionRouter(nn.Module):
    def __init__(self, cfg, d_model: int):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        token2sent: torch.Tensor,
        is_sent_head: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        hidden: [B, N, D] after mid layers
        We only route over sentence-head tokens.
        """
        # Gather sentence head representations.
        # Note: this is a stub; in practice you'll pack heads into a dense tensor.
        _head_mask = is_sent_head > 0.0  # [B, N]
        # TODO: implement dense extraction of head reps.

        # Placeholder to keep the interface; you'll fill the real logic.
        logits = torch.zeros_like(is_sent_head)
        probs = torch.sigmoid(logits)
        is_sec_head = (probs > 0.5).float() + probs - probs.detach()

        token2sec = torch.zeros_like(token2sent)

        return {
            "logits": logits,
            "probs": probs,
            "is_sec_head": is_sec_head,
            "token2sec": token2sec.long(),
        }
