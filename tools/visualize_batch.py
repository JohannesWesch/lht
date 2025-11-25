#!/usr/bin/env python3
"""
Utility to visualize attention patterns on a single batch.

Useful for:
- Debugging hierarchical attention
- Creating figures for papers/presentations
- Quick sanity checks on attention behavior
- Understanding ML-SWA cascading windows

Usage:
    python tools/visualize_batch.py --config configs/pretrain_hierarchical.yaml
    python tools/visualize_batch.py --checkpoint checkpoints/my-model/checkpoint-10000.ckpt
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from lht.config import load_config
from lht.lightning_module import LHTLightningModule
from lht.utils.nested_builder import build_coords_from_nested_list
from lht.visualization import compute_attention_entropy, create_attention_heatmap


def visualize_single_batch(
    config_path: str = None,
    checkpoint_path: str = None,
    output_path: str = "attention_visualization.png",
    max_tokens: int = 60,
):
    """
    Visualize attention patterns on a sample hierarchical document.

    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to trained checkpoint (optional)
        output_path: Where to save the heatmap
        max_tokens: Maximum tokens to visualize
    """
    print("=" * 80)
    print("SINGLE BATCH ATTENTION VISUALIZATION")
    print("=" * 80)
    print()

    # Load config
    if config_path:
        print(f"Loading config from: {config_path}")
        cfg = load_config(config_path)
    elif checkpoint_path:
        print(f"Loading config from checkpoint: {checkpoint_path}")
        # Load model and extract config
        model = LHTLightningModule.load_from_checkpoint(checkpoint_path)
        cfg = model.config
    else:
        raise ValueError("Must provide either config_path or checkpoint_path")

    print(f"  Model: {cfg.model.d_model}d, {cfg.model.num_layers} layers")
    print()

    # Initialize model
    print("Initializing model...")
    if checkpoint_path:
        model = LHTLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=cfg,
            strict=False,
        )
        print(f"  ✓ Loaded from checkpoint: {checkpoint_path}")
    else:
        model = LHTLightningModule(cfg)
        print("  ✓ Created fresh model (random weights)")

    model.eval()
    print()

    # Create sample document
    print("Creating sample document...")
    document = [
        # Section 1: Introduction
        [
            "Transformer models have revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant context.",
            "However, long sequences remain computationally challenging.",
        ],
        # Section 2: Methods
        [
            "We propose a hierarchical sliding window attention mechanism.",
            "Our approach operates at token, sentence, and section levels.",
            "This enables efficient processing of long documents.",
        ],
        # Section 3: Results
        [
            "Experiments show significant improvements in efficiency.",
            "The model maintains strong performance on downstream tasks.",
        ],
    ]

    print("  Document structure:")
    for sec_idx, section in enumerate(document):
        print(f"    Section {sec_idx + 1}: {len(section)} sentences")
    print()

    # Tokenize and build positions
    print("Tokenizing and building hierarchical positions...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    input_ids, positions = build_coords_from_nested_list(
        document=document,
        tokenizer=tokenizer,
        max_length=512,
        device=torch.device("cpu"),
    )

    seq_len = len(input_ids)
    print(f"  ✓ Sequence length: {seq_len} tokens")

    # Decode tokens for visualization
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    # Count boundaries
    sentence_boundaries = (positions.level_enums[1] > 0).sum().item()
    section_boundaries = (positions.level_enums[2] > 0).sum().item()
    print(
        f"  ✓ Detected {sentence_boundaries} sentence boundaries, {section_boundaries} section boundaries"
    )
    print()

    # Create batch
    batch = {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.ones(1, len(input_ids)),
        "positions": positions,
    }

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            positions=batch["positions"],
        )

    print("  ✓ Forward pass completed")
    print()

    # Check if model returns attention weights
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        print("Creating attention heatmap...")

        # Take first layer attention
        attn_weights = outputs.attentions[0][0]  # [H, N, N]

        # Limit to max_tokens
        display_len = min(seq_len, max_tokens)
        attn_subset = attn_weights[:, :display_len, :display_len]
        tokens_subset = tokens[:display_len]

        # Create heatmap
        fig = create_attention_heatmap(
            attention_weights=attn_subset,
            tokens=tokens_subset,
            layer_name="ML-SWA Layer 0",
            max_tokens=display_len,
        )

        # Save
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved attention heatmap to: {output_path}")
        print()

        # Compute entropy statistics
        print("Analyzing attention entropy...")
        entropy = compute_attention_entropy(attn_weights.unsqueeze(0))

        print(f"  Mean entropy: {entropy.mean().item():.4f}")
        print(f"  Std entropy:  {entropy.std().item():.4f}")
        print(f"  Min entropy:  {entropy.min().item():.4f} (most focused)")
        print(f"  Max entropy:  {entropy.max().item():.4f} (most diffuse)")
        print()

        # Analyze boundary vs non-boundary
        sent_boundary_mask = positions.level_enums[1] > 0
        if sent_boundary_mask.any():
            boundary_entropy = entropy[0, 0, sent_boundary_mask].mean().item()
            non_boundary_entropy = entropy[0, 0, ~sent_boundary_mask].mean().item()

            print("  Boundary vs Regular tokens:")
            print(f"    Sentence boundaries: {boundary_entropy:.4f}")
            print(f"    Regular tokens:      {non_boundary_entropy:.4f}")
            if boundary_entropy > non_boundary_entropy:
                print("    → Boundaries attend more broadly")
            else:
                print("    → Boundaries attend more focused")
        print()
    else:
        print("⚠ Model does not return attention weights")
        print(
            "  To enable attention visualization, modify your model's forward method to return attention_weights"
        )
        print()
        print("  In src/lht/model.py, add:")
        print("    return {")
        print('        "mlm_logits": logits,')
        print('        "hidden": x,')
        print('        "attentions": attention_weights,  # Add this line')
        print("    }")
        print()

    # Show model outputs
    print("Model output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:15s}: {tuple(value.shape)}")
    print()

    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        print(f"✓ Attention heatmap saved to: {output_path}")
        print()
        print("The heatmap shows:")
        print("  • Diagonal bands: local token-level attention")
        print("  • Cross-connections: sentence/section boundary attention")
        print("  • Cascading windows across hierarchy levels")
    else:
        print(
            "To visualize attention patterns, modify your model to return attention weights."
        )
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns on a single batch"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attention_visualization.png",
        help="Output path for heatmap (default: attention_visualization.png)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=60,
        help="Maximum tokens to visualize (default: 60)",
    )

    args = parser.parse_args()

    if not args.config and not args.checkpoint:
        parser.error("Must provide either --config or --checkpoint")

    visualize_single_batch(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
