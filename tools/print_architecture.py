#!/usr/bin/env python3
"""
Print detailed architecture layout of the LHT model.
Shows per-layer configuration including hierarchy levels and window sizes.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lht.config import load_config
from lht.model import LHTEncoder


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_architecture(config_path: str):
    """Print detailed architecture layout."""
    print("=" * 80)
    print("LHT ENCODER ARCHITECTURE")
    print("=" * 80)

    # Load config
    cfg = load_config(config_path)
    model_cfg = cfg.model

    # Print model config
    print("\n📊 MODEL CONFIGURATION")
    print(f"  Vocabulary Size:     {model_cfg.vocab_size:,}")
    print(f"  Model Dimension:     {model_cfg.d_model}")
    print(f"  Number of Layers:    {model_cfg.num_layers}")
    print(f"  Number of Heads:     {model_cfg.n_heads}")
    print(f"  Head Dimension:      {model_cfg.d_model // model_cfg.n_heads}")
    print(f"  Feedforward Dim:     {model_cfg.d_ff}")
    print(f"  Dropout:             {model_cfg.dropout}")
    print(f"  Max Sequence Length: {model_cfg.max_seq_len:,}")

    # Print ML-SWA config
    if model_cfg.mlswa:
        print("\n🔍 MULTI-LEVEL SLIDING WINDOW ATTENTION (ML-SWA)")
        print("  Window Sizes (per level):")
        for i, window_size in enumerate(model_cfg.mlswa.window_size_per_level):
            level_names = ["Token", "Sentence", "Section", "Chapter"]
            level_name = level_names[i] if i < len(level_names) else f"Level {i}"
            print(f"    Level {i} ({level_name:8s}): {window_size:4d} positions")

        print("\n  Per-Layer Hierarchy Levels:")
        layer_max_levels = model_cfg.mlswa.layer_max_level

        for layer_idx in range(model_cfg.num_layers):
            if layer_idx < len(layer_max_levels):
                max_level = layer_max_levels[layer_idx]
            else:
                max_level = None

            if max_level is None:
                active_levels = list(range(len(model_cfg.mlswa.window_size_per_level)))
                level_str = "All levels"
            else:
                active_levels = list(range(max_level + 1))
                level_str = f"Levels 0-{max_level}"

            # Calculate effective attention span
            max_span = max(
                model_cfg.mlswa.window_size_per_level[i] for i in active_levels
            )

            print(
                f"    Layer {layer_idx:2d}: {level_str:15s} "
                f"(max span: {max_span:4d} positions)"
            )

    # Initialize model to count parameters
    print("\n⚙️  PARAMETER BREAKDOWN")
    model = LHTEncoder(cfg)
    trainable, total = count_parameters(model)

    print(f"  Total Parameters:      {total:,}")
    print(f"  Trainable Parameters:  {trainable:,}")
    print(f"  Model Size:            {total * 4 / (1024**2):.2f} MB (fp32)")

    # Break down by component
    print("\n  Component-wise parameters:")

    # Embeddings
    embed_params = sum(p.numel() for p in model.token_embed.parameters())
    print(
        f"    Token Embeddings:    {embed_params:,} " f"({embed_params/total*100:.1f}%)"
    )

    # Transformer layers
    layer_params = sum(p.numel() for p in model.layers.parameters())
    avg_layer_params = layer_params / model_cfg.num_layers
    print(
        f"    All Transformer Layers: {layer_params:,} "
        f"({layer_params/total*100:.1f}%)"
    )
    print(f"      Per-layer average:    {avg_layer_params:,.0f}")

    # Single layer breakdown (first layer as example)
    if len(model.layers) > 0:
        layer_0 = model.layers[0]
        qkv_params = sum(p.numel() for p in layer_0.qkv.parameters())
        out_proj_params = sum(p.numel() for p in layer_0.out_proj.parameters())
        ff_params = sum(p.numel() for p in layer_0.ff.parameters())
        norm_params = sum(p.numel() for p in layer_0.norm1.parameters()) + sum(
            p.numel() for p in layer_0.norm2.parameters()
        )

        print("      └─ Per-layer breakdown:")
        print(f"         QKV Projections:    {qkv_params:,}")
        print(f"         Output Projection:  {out_proj_params:,}")
        print(f"         Feedforward Net:    {ff_params:,}")
        print(f"         Layer Norms:        {norm_params:,}")

    # Final norm
    final_norm_params = sum(p.numel() for p in model.final_norm.parameters())
    print(f"    Final Layer Norm:    {final_norm_params:,}")

    # MLM head
    mlm_params = sum(p.numel() for p in model.mlm_head.parameters())
    print(f"    MLM Head:            {mlm_params:,} " f"({mlm_params/total*100:.1f}%)")

    # Attention pattern info
    print("\n💡 ATTENTION MECHANICS")
    print("  Type: Multi-Level Sliding Window (Sparse)")
    print("  Computation: Lazy evaluation (O(L×N) memory)")
    print("  Implementation: FlexAttention with loop-unrolled mask")
    print("  Vmap-safe: Yes")
    print("  Scales to: 128k+ tokens")

    print("\n🎯 ATTENTION PATTERN PROPERTIES")
    print("  • Token level (L0): All tokens attend within local window")
    print("  • Sentence level (L1): Boundary tokens attend to sentence boundaries")
    print("  • Section level (L2): Boundary tokens attend to section boundaries")
    print("  • Merging: OR (union) across levels")
    print("  • Distance: Enumeration-space (not physical positions)")

    print("\n" + "=" * 80)
    print("Note: Sliding window patterns are NOT trainable parameters.")
    print("      They are computed on-the-fly based on hierarchical positions.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Print detailed LHT architecture layout"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_hierarchical.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    print_architecture(args.config)


if __name__ == "__main__":
    main()
