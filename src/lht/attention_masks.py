"""
xFormers attention bias utilities.

This module provides AttentionBias implementations for:
- local sliding window attention,
- sentence-aware attention (children + parents + neighbours),
- full hierarchical attention (HDT-style child/parent/sibling).

You'll fill in the actual boolean logic for the masks.
"""

import torch
from xformers.ops.fmha.attn_bias import AttentionBias, LocalAttentionFromBottomRightMask


def build_local_attention_bias(
    seq_len: int,
    window_size: int,
    device: torch.device,
) -> AttentionBias:
    """
    Build a sliding-window attention bias for local layers.

    This uses xFormers' built-in local mask.
    """
    return LocalAttentionFromBottomRightMask(
        window_left=window_size,
        window_right=window_size,
        seq_len_q=seq_len,
        seq_len_k=seq_len,
        device=device,
    )


class SentenceAwareBias(AttentionBias):
    """
    Attention bias that restricts tokens primarily to their sentence,
    sentence head, and optional [DOC] token.

    token2sent: [B, N]
    is_sent_head: [B, N]
    """

    def __init__(
        self,
        token2sent: torch.Tensor,
        is_sent_head: torch.Tensor,
        use_doc_token: bool = True,
    ) -> None:
        super().__init__()
        self.token2sent = token2sent
        self.is_sent_head = is_sent_head
        self.use_doc_token = use_doc_token

    def materialize(
        self,
        shape,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """
        Returns a bias tensor of shape [B, 1, Nq, Nk] with 0.0 for allowed pairs
        and -inf for disallowed ones. You'll implement the actual logic.
        """
        # shape: (B, H, Nq, Nk) or (B, Nq, Nk); here we assume (B, N, N)
        _B, Nq, Nk = shape[-3], shape[-2], shape[-1]

        token2sent = self.token2sent.to(device)
        q_sent = token2sent[:, :Nq].unsqueeze(-1)  # [B, Nq, 1]
        k_sent = token2sent[:, :Nk].unsqueeze(-2)  # [B, 1, Nk]

        same_sent = q_sent == k_sent  # [B, Nq, Nk]

        # Start with: only same sentence allowed.
        mask = same_sent

        # TODO: add parent (head) & [DOC] token logic here.

        bias = torch.zeros_like(mask, dtype=dtype or torch.float32)
        bias = bias.masked_fill(~mask, float("-inf"))
        return bias.unsqueeze(1)  # [B, 1, Nq, Nk]


class HierarchicalBias(AttentionBias):
    """
    Generic HDT-style hierarchical attention for K abstraction levels.

    Encodes child/parent/sibling relationships across arbitrary hierarchy depth:
    - 2 levels: tokens ↔ sentences ↔ [DOC]
    - 3 levels: tokens ↔ sentences ↔ sections ↔ [DOC]
    - 4 levels: tokens ↔ sentences ↔ paragraphs ↔ sections ↔ [DOC]
    - etc.

    level_ids: List of [B, N] tensors, one per level ℓ = 1..K
    is_heads: List of [B, N] tensors, head flags per level
    """

    def __init__(
        self,
        level_ids: list,  # List[Tensor[B, N]] of length K
        is_heads: list,  # List[Tensor[B, N]] of length K
        use_doc_token: bool,
        neighbour_sentences: int = 1,
        neighbour_sections: int = 1,
    ) -> None:
        super().__init__()
        self.level_ids = level_ids  # level_ids[ℓ] for ℓ = 0..K-1
        self.is_heads = is_heads  # is_heads[ℓ] for ℓ = 0..K-1
        self.num_levels = len(level_ids)
        self.use_doc_token = use_doc_token
        self.neighbour_sentences = neighbour_sentences
        self.neighbour_sections = neighbour_sections

        # Backward compatibility: expose first two levels with old names
        if self.num_levels >= 1:
            self.token2sent = level_ids[0]
            self.is_sent_head = is_heads[0]
        if self.num_levels >= 2:
            self.token2sec = level_ids[1]
            self.is_sec_head = is_heads[1]

    def materialize(
        self,
        shape,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """
        Returns a bias tensor implementing the child/parent/sibling attention
        pattern for arbitrary K levels.

        For each level ℓ = 1..K:
        - Child edges: tokens in same unit at level ℓ
        - Parent edges: tokens to their head at level ℓ
        - Sibling edges: units that share a parent at level ℓ+1

        This is the generic HDT-like pattern.
        """
        _B, Nq, Nk = shape[-3], shape[-2], shape[-1]
        device = device or self.level_ids[0].device
        dtype = dtype or torch.float32

        # Start with no edges allowed
        mask = torch.zeros((shape[-3], Nq, Nk), dtype=torch.bool, device=device)

        # Add edges for each level in the hierarchy
        for level_idx, level_id in enumerate(self.level_ids):
            level_id = level_id.to(device)

            # Same-unit edges at this level (children)
            q_level = level_id[:, :Nq].unsqueeze(-1)  # [B, Nq, 1]
            k_level = level_id[:, :Nk].unsqueeze(-2)  # [B, 1, Nk]
            same_unit = q_level == k_level  # [B, Nq, Nk]

            mask = mask | same_unit

            # TODO: Add parent edges (token → head at each level)
            # TODO: Add sibling edges (units sharing parent at level+1)
            # TODO: Add neighbour edges with windowing

        # TODO: Add [DOC] token edges if use_doc_token is True

        bias = torch.zeros_like(mask.float(), dtype=dtype)
        bias = bias.masked_fill(~mask, float("-inf"))
        return bias.unsqueeze(1)  # [B, 1, Nq, Nk]
