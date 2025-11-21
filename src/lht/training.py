"""
Training loop utilities for LHT.

This module defines:
- MLM (Masked Language Modeling) utilities matching HDT's setup
- Router ratio loss computation
- Training step implementation
- Logging hooks
"""

from typing import Dict, Tuple

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
    Single training step for LHT with MLM objective + router losses.

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

    # TODO: Add LM head to get logits from hidden states
    # For now, this is a placeholder
    # mlm_loss = compute_mlm_loss(logits, labels)

    # Router losses from hierarchy state (schedule-driven)
    hier_state = outputs["hierarchy"]
    router_ratio_loss = outputs["router_ratio_loss"]

    # Total loss (TODO: add MLM loss once LM head is implemented)
    # total_loss = mlm_loss + router_ratio_loss
    total_loss = router_ratio_loss  # placeholder

    # Compute statistics for logging
    with torch.no_grad():
        _level_ids = hier_state.get("level_ids", {})
        is_heads = hier_state.get("is_heads", {})
        ratio_losses = hier_state.get("ratio_losses", {})

        stats = {}
        if attention_mask is not None:
            num_valid = attention_mask.sum()
            for level_num, is_head in is_heads.items():
                level_name = cfg.hierarchy.levels[level_num - 1].name
                num_heads = (is_head * attention_mask).sum()
                stats[f"avg_{level_name}s_per_doc"] = (num_heads / num_valid).item()
                stats[f"compression_{level_name}"] = (
                    num_valid / num_heads.clamp(min=1)
                ).item()

    # Build result dict
    result = {
        "loss": total_loss,
        "router_ratio_loss": router_ratio_loss,
        # "mlm_loss": mlm_loss,  # TODO: add when LM head exists
    }

    # Add per-level losses
    for level_num, loss in ratio_losses.items():
        level_name = cfg.hierarchy.levels[level_num - 1].name
        result[f"ratio_loss_{level_name}"] = loss

    result.update(stats)

    return result


# (Main training loop to be added here.)
