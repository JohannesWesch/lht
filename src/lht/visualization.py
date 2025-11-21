"""
Visualization utilities for W&B logging.

This module provides functions for:
- Sample MLM predictions with decoded text
- Attention heatmap visualization
- Hierarchy boundary overlays
- Gradient flow visualization
- Attention entropy computation
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb


def log_sample_predictions(
    input_ids: torch.Tensor,
    masked_input_ids: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    tokenizer,
    num_samples: int = 3,
    top_k: int = 5,
) -> wandb.Table:
    """
    Log MLM sample predictions to wandb as a table.

    Shows original text, masked text, predicted tokens, and whether they match.

    Args:
        input_ids: [B, N] original token IDs
        masked_input_ids: [B, N] masked token IDs
        labels: [B, N] with -100 for unmasked positions
        logits: [B, N, vocab_size] model predictions
        tokenizer: HuggingFace tokenizer
        num_samples: number of sequences to show
        top_k: show top-k predictions per masked token

    Returns:
        wandb.Table with predictions
    """
    table = wandb.Table(
        columns=[
            "Sample",
            "Position",
            "Original",
            "Masked",
            "Top-5 Predictions",
            "Correct",
        ]
    )

    B, N = input_ids.shape
    num_samples = min(num_samples, B)

    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)  # [B, N, V]
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)  # [B, N, k]

        for sample_idx in range(num_samples):
            # Find masked positions
            masked_positions = (labels[sample_idx] != -100).nonzero(as_tuple=True)[0]

            for pos in masked_positions[:10]:  # limit to 10 masked tokens per sample
                pos = pos.item()

                # Decode tokens
                original_token = tokenizer.decode([input_ids[sample_idx, pos].item()])
                masked_token = tokenizer.decode(
                    [masked_input_ids[sample_idx, pos].item()]
                )

                # Get top-k predictions
                predictions = []
                for k_idx in range(top_k):
                    pred_id = top_indices[sample_idx, pos, k_idx].item()
                    pred_token = tokenizer.decode([pred_id])
                    pred_prob = top_probs[sample_idx, pos, k_idx].item()
                    predictions.append(f"{pred_token} ({pred_prob:.2%})")

                predictions_str = ", ".join(predictions)

                # Check if prediction is correct
                pred_id = top_indices[sample_idx, pos, 0].item()
                correct = pred_id == input_ids[sample_idx, pos].item()

                table.add_data(
                    sample_idx,
                    pos,
                    original_token,
                    masked_token,
                    predictions_str,
                    "✓" if correct else "✗",
                )

    return table


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: Optional[List[str]] = None,
    layer_name: str = "Layer",
    max_tokens: int = 128,
) -> plt.Figure:
    """
    Create attention heatmap visualization.

    Args:
        attention_weights: [H, N, N] or [N, N] attention weights
        tokens: optional list of token strings for axis labels
        layer_name: name for the plot title
        max_tokens: maximum tokens to display (for readability)

    Returns:
        matplotlib Figure
    """
    # Handle multi-head attention
    if attention_weights.dim() == 3:
        # Average over heads
        attn = attention_weights.mean(dim=0).detach().cpu().numpy()
    else:
        attn = attention_weights.detach().cpu().numpy()

    # Limit size for visualization
    N = min(attn.shape[0], max_tokens)
    attn = attn[:N, :N]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        attn,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Attention Weight"},
        ax=ax,
        vmin=0,
        vmax=attn.max(),
    )

    ax.set_title(f"{layer_name} Attention Pattern", fontsize=14, fontweight="bold")
    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)

    # Add token labels if provided
    if tokens is not None and len(tokens) >= N:
        tick_labels = tokens[:N]
        ax.set_xticks(np.arange(N) + 0.5)
        ax.set_yticks(np.arange(N) + 0.5)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)

    plt.tight_layout()
    return fig


def visualize_hierarchy_boundaries(
    text: str,
    boundary_positions: Dict[str, List[int]],
    tokenizer,
) -> plt.Figure:
    """
    Visualize hierarchy boundaries overlaid on text.

    Args:
        text: input text string
        boundary_positions: dict mapping level_name -> list of boundary token positions
        tokenizer: HuggingFace tokenizer

    Returns:
        matplotlib Figure
    """
    tokens = tokenizer.tokenize(text)
    n_tokens = len(tokens)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot tokens
    x_positions = np.arange(n_tokens)
    y_base = 0

    # Color map for different levels
    level_colors = {
        "sentence": "blue",
        "paragraph": "green",
        "section": "red",
    }

    # Draw boundaries for each level
    for _level_idx, (level_name, boundaries) in enumerate(boundary_positions.items()):
        color = level_colors.get(level_name, "gray")

        for boundary_pos in boundaries:
            if boundary_pos < n_tokens:
                ax.axvline(
                    x=boundary_pos,
                    ymin=y_base,
                    ymax=y_base + 0.25,
                    color=color,
                    linewidth=2,
                    label=level_name if boundary_pos == boundaries[0] else "",
                    alpha=0.7,
                )

    # Add token labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_ylim(-0.1, 1)
    ax.set_xlim(-1, n_tokens)
    ax.set_title("Learned Hierarchy Boundaries", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def compute_attention_entropy(
    attention_weights: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute entropy of attention distributions.

    Higher entropy = more diffuse attention.
    Lower entropy = more focused attention.

    Args:
        attention_weights: [B, H, N, N] attention weights
        attention_mask: [B, N] optional mask

    Returns:
        entropy: [B, H, N] entropy per query position
    """
    B, H, N, _ = attention_weights.shape

    # Apply mask if provided
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attention_weights = attention_weights.masked_fill(~mask.bool(), 1e-10)

    # Normalize to ensure valid probability distribution
    attn_probs = attention_weights / (
        attention_weights.sum(dim=-1, keepdim=True) + 1e-10
    )

    # Compute entropy: -sum(p * log(p))
    log_probs = torch.log(attn_probs + 1e-10)
    entropy = -torch.sum(attn_probs * log_probs, dim=-1)  # [B, H, N]

    return entropy


def log_gradient_flow(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
) -> plt.Figure:
    """
    Visualize gradient magnitudes across layers.

    Args:
        named_parameters: list of (name, parameter) tuples from model.named_parameters()

    Returns:
        matplotlib Figure showing gradient flow
    """
    # Collect gradient norms
    layer_names = []
    grad_norms = []

    for name, param in named_parameters:
        if param.grad is not None and "embed" not in name:  # skip embeddings
            layer_names.append(name)
            grad_norms.append(param.grad.norm().item())

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot gradient norms
    x_positions = np.arange(len(layer_names))
    bars = ax.bar(x_positions, grad_norms, color="steelblue", alpha=0.7)

    # Highlight layers with very small or large gradients
    for i, norm in enumerate(grad_norms):
        if norm < 1e-5:
            bars[i].set_color("red")  # vanishing gradient
        elif norm > 10:
            bars[i].set_color("orange")  # exploding gradient

    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_names, rotation=90, fontsize=8)
    ax.set_ylabel("Gradient Norm", fontsize=12)
    ax.set_title("Gradient Flow Across Layers", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def log_router_statistics(
    hierarchy_state: Dict,
    attention_mask: torch.Tensor,
    config,
) -> Dict[str, float]:
    """
    Compute detailed router statistics for logging.

    Args:
        hierarchy_state: hierarchy state dict from model forward pass
        attention_mask: [B, N] attention mask
        config: experiment config

    Returns:
        dict of statistics
    """
    stats = {}

    level_ids = hierarchy_state.get("level_ids", {})
    is_heads = hierarchy_state.get("is_heads", {})
    ratio_losses = hierarchy_state.get("ratio_losses", {})

    num_valid = attention_mask.sum()

    for level_num, level_id_tensor in level_ids.items():
        level_name = config.hierarchy.levels[level_num - 1].name
        is_head = is_heads[level_num]

        # Count heads
        num_heads = (is_head * attention_mask).sum().item()
        avg_heads_per_doc = num_heads / attention_mask.size(0)

        # Compression ratio
        compression = num_valid.item() / max(num_heads, 1)

        # Ratio loss
        ratio_loss = ratio_losses.get(level_num, torch.tensor(0.0)).item()

        # Number of unique groups
        num_groups = len(torch.unique(level_id_tensor[attention_mask.bool()]))

        stats[f"router/{level_name}_heads_per_doc"] = avg_heads_per_doc
        stats[f"router/{level_name}_compression"] = compression
        stats[f"router/{level_name}_ratio_loss"] = ratio_loss
        stats[f"router/{level_name}_num_groups"] = num_groups

        # Boundary density (heads per 100 tokens)
        boundary_density = (num_heads / num_valid.item()) * 100
        stats[f"router/{level_name}_boundary_density_pct"] = boundary_density

    return stats
