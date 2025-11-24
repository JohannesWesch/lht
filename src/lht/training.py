"""
Training loop utilities for LHT.

This module defines:
- MLM (Masked Language Modeling) loss computation
"""

import torch


def compute_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked language modeling loss.

    Args:
        logits: [B, N, vocab_size] model predictions
        labels: [B, N] with -100 for positions to ignore

    Returns:
        loss: scalar
    """
    # CrossEntropyLoss with default reduction='mean' averages over all valid (non -100) positions
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    # Flatten for loss computation
    shift_logits = logits.view(-1, logits.size(-1))
    shift_labels = labels.view(-1)

    # Compute loss
    loss = loss_fct(shift_logits, shift_labels)
    return loss
