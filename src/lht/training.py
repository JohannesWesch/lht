"""
Training loop utilities for LHT.

This module defines:
- a Trainer-like class or simple functions to run pretraining,
- computation of router ratio losses,
- logging hooks.
"""

import torch


def compute_ratio_loss(
    probs: torch.Tensor,
    target_ratio: float,
    weight: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage router probabilities to yield a specific proportion of heads.

    probs: [B, N]
    mask: [B, N] (1 for valid tokens, 0 for padding)
    """
    num_valid = mask.sum(dim=-1).clamp(min=1.0)
    expected_heads = (probs * mask).sum(dim=-1) / num_valid  # [B]
    loss = (expected_heads - target_ratio).abs().mean()
    return weight * loss


# (You'll add the main training loop here later.)
