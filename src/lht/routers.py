"""
Generic routers for learning hierarchical boundaries at any level.

This module implements H-Net-style routing generalized to K levels:
- LevelRouter: predicts head tokens at any abstraction level
- Uses straight-through estimation (STE) for gradients through discrete decisions
- Works for: token→sentence, sentence→paragraph, paragraph→section, etc.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class LevelRouter(nn.Module):
    """
    Generic router for predicting boundaries at any hierarchical level.

    Can operate on:
    - Tokens (level 0 → level 1, e.g., token → sentence)
    - Previous-level heads (level ℓ-1 → level ℓ, e.g., sentence → section)

    Uses straight-through estimation (STE) to allow gradient flow through
    discrete boundary decisions.
    """

    def __init__(self, level_name: str, d_model: int, router_config):
        super().__init__()
        self.level_name = level_name
        self.d_model = d_model
        self.cfg = router_config

        hidden = router_config.hidden_dim

        # Simple MLP for boundary prediction
        # Can be enhanced with convolution for local context
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        prev_level_ids: Optional[torch.Tensor] = None,
        prev_is_heads: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict boundaries at this level.

        Args:
            hidden: [B, N, D] representations (tokens or previous-level units)
            prev_level_ids: [B, N] IDs from previous level (optional)
            prev_is_heads: [B, N] head flags from previous level (optional)
            attention_mask: [B, N] valid token mask

        Returns:
            dict with:
                - logits: [B, N] boundary logits
                - probs: [B, N] boundary probabilities
                - is_head: [B, N] binary (STE) head flags
                - level_ids: [B, N] unit IDs at this level
        """
        _batch_size, _seq_len, _ = hidden.shape

        # For first-level router (tokens → level 1):
        # Operate directly on token representations
        #
        # For higher-level routers (level ℓ-1 → level ℓ):
        # Should ideally extract head representations from prev level
        # For now, still use full hidden; TODO: implement head extraction

        if prev_is_heads is not None:
            # Higher-level router: could extract head representations
            # _head_mask = prev_is_heads > 0.0
            # TODO: implement dense head extraction and routing
            # For now, route over all positions (will be improved)
            pass

        # Predict boundary probabilities
        logits = self.mlp(hidden).squeeze(-1)  # [B, N]
        probs = torch.sigmoid(logits)

        # Straight-through estimation (STE) for binary decisions
        # Forward: use hard threshold
        # Backward: use soft probabilities
        hard = (probs > 0.5).float()
        is_head = hard + probs - probs.detach()

        # Assign unit IDs via cumulative sum of heads
        # Each head marks the start of a new unit
        level_ids = torch.cumsum(is_head, dim=-1) - 1  # [0, 0, 1, 1, 1, 2, ...]

        # Mask out invalid positions
        if attention_mask is not None:
            level_ids = level_ids * attention_mask + (-1) * (1 - attention_mask)

        return {
            "logits": logits,
            "probs": probs,
            "is_head": is_head,
            "level_ids": level_ids.long(),
        }


# Legacy aliases for backward compatibility
class TokenToSentenceRouter(LevelRouter):
    """Alias for first-level router (tokens → sentences)."""

    def __init__(self, cfg, d_model: int):
        super().__init__(level_name="sentence", d_model=d_model, router_config=cfg)

    def forward(
        self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(hidden, attention_mask=attention_mask)
        # Add legacy key names
        out["is_sent_head"] = out["is_head"]
        out["token2sent"] = out["level_ids"]
        return out


class SentenceToSectionRouter(LevelRouter):
    """Alias for second-level router (sentences → sections)."""

    def __init__(self, cfg, d_model: int):
        super().__init__(level_name="section", d_model=d_model, router_config=cfg)

    def forward(
        self,
        hidden: torch.Tensor,
        token2sent: torch.Tensor,
        is_sent_head: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(
            hidden, prev_level_ids=token2sent, prev_is_heads=is_sent_head
        )
        # Add legacy key names
        out["is_sec_head"] = out["is_head"]
        out["token2sec"] = out["level_ids"]
        return out
