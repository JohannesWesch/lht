"""
Training loop utilities for LHT.

This module defines:
- MLM (Masked Language Modeling) utilities
- Training step implementation
- Logging hooks
"""

from typing import Dict, Tuple

import torch


def mask_tokens_for_mlm(
    input_ids: torch.Tensor,
    tokenizer,
    mlm_probability: float = 0.15,
    special_token_ids: set = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens for MLM following BERT-style masking.

    HDT setup:
    - 15% of tokens are selected for masking
    - Of selected tokens: 80% -> [MASK], 10% -> random, 10% -> unchanged
    - Special tokens ([DOC], [SEC], [CLS], [PAD]) are never masked

    Args:
        input_ids: [B, N] token IDs
        tokenizer: HuggingFace tokenizer
        mlm_probability: probability of masking (default 0.15)
        special_token_ids: set of token IDs to never mask

    Returns:
        masked_input_ids: [B, N] with masks applied
        labels: [B, N] with -100 for non-masked positions
    """
    labels = input_ids.clone()

    # Get special token IDs that should never be masked
    if special_token_ids is None:
        special_token_ids = set(tokenizer.all_special_ids)

    # Create probability matrix
    probability_matrix = torch.full(input_ids.shape, mlm_probability)

    # Don't mask special tokens
    special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in special_token_ids:
        special_tokens_mask |= input_ids == special_id

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Sample which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time: replace with [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    )
    masked_input_ids = input_ids.clone()
    masked_input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time: replace with random token
    indices_random = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    masked_input_ids[indices_random] = random_words[indices_random]

    # 10% of the time: keep unchanged (already in masked_input_ids)

    return masked_input_ids, labels


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


def training_step(
    model,
    batch: Dict[str, torch.Tensor],
    tokenizer,
    cfg,
) -> Dict[str, torch.Tensor]:
    """
    Single training step for LHT with MLM objective.

    Args:
        model: LHTEncoder
        batch: dict with 'input_ids' and 'attention_mask'
        tokenizer: tokenizer for masking
        cfg: training config

    Returns:
        dict with losses and diagnostics
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Create MLM masks
    masked_input_ids, labels = mask_tokens_for_mlm(
        input_ids,
        tokenizer,
        mlm_probability=cfg.training.mlm_probability,
    )

    # Forward pass
    outputs = model(masked_input_ids, attention_mask=attention_mask)

    # Compute MLM loss from model's mlm_logits
    mlm_logits = outputs["mlm_logits"]
    mlm_loss = compute_mlm_loss(mlm_logits, labels)

    # Total loss
    total_loss = mlm_loss

    # Build result dict
    result = {
        "loss": total_loss,
        "mlm_loss": mlm_loss,
    }

    return result
