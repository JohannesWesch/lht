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
    Full HDT-style hierarchical attention.

    Encodes:
    - token -> sentence (children/parent/neighbours),
    - token -> section,
    - sentence head -> section head,
    - section head -> [DOC].

    token2sent: [B, N]
    token2sec: [B, N]
    is_sent_head: [B, N]
    is_sec_head: [B, N]
    """

    def __init__(
        self,
        token2sent: torch.Tensor,
        token2sec: torch.Tensor,
        is_sent_head: torch.Tensor,
        is_sec_head: torch.Tensor,
        use_doc_token: bool,
        neighbour_sentences: int,
        neighbour_sections: int,
    ) -> None:
        super().__init__()
        self.token2sent = token2sent
        self.token2sec = token2sec
        self.is_sent_head = is_sent_head
        self.is_sec_head = is_sec_head
        self.use_doc_token = use_doc_token
        self.neighbour_sentences = neighbour_sentences
        self.neighbour_sections = neighbour_sections

    def materialize(
        self,
        shape,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        """
        Returns a bias tensor implementing the child/parent/sibling attention
        pattern. This is where you reproduce HDT-like semantics.
        """
        _B, Nq, Nk = shape[-3], shape[-2], shape[-1]
        device = device or self.token2sent.device
        dtype = dtype or torch.float32

        token2sent = self.token2sent.to(device)
        token2sec = self.token2sec.to(device)

        # Basic same-sentence baseline.
        q_sent = token2sent[:, :Nq].unsqueeze(-1)  # [B, Nq, 1]
        k_sent = token2sent[:, :Nk].unsqueeze(-2)  # [B, 1, Nk]
        same_sent = q_sent == k_sent  # [B, Nq, Nk]

        # Same-section baseline.
        q_sec = token2sec[:, :Nq].unsqueeze(-1)
        k_sec = token2sec[:, :Nk].unsqueeze(-2)
        same_sec = q_sec == k_sec

        mask = same_sent | same_sec

        # TODO: add neighbour sentences/sections, head-child, [DOC] logic.

        bias = torch.zeros_like(mask, dtype=dtype)
        bias = bias.masked_fill(~mask, float("-inf"))
        return bias.unsqueeze(1)  # [B, 1, Nq, Nk]
