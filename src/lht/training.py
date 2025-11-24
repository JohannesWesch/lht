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
    loss_fct = torch.nn.CrossEntropyLoss()
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
