#!/usr/bin/env python3
"""
Inspect hierarchical enumeration vectors from the dataloader.
Shows the three hierarchy levels for the first example from unarxive.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lht.config import load_config
from lht.utils.nested_builder import build_coords_from_nested_list


def visualize_hierarchy_vectors(
    config_path: str,
    max_tokens_display: int = 100,
    show_boundaries_only: bool = False,
    use_test_document: bool = True,
):
    """
    Visualize hierarchy vectors on a test document or from dataloader.

    Args:
        config_path: Path to config file
        max_tokens_display: Maximum tokens to display
        show_boundaries_only: If True, only show boundary tokens
        use_test_document: If True, use hardcoded test document
    """
    print("=" * 100)
    print(
        "HIERARCHICAL ENUMERATION VECTORS - TEST DOCUMENT"
        if use_test_document
        else "HIERARCHICAL ENUMERATION VECTORS - FIRST EXAMPLE"
    )
    print("=" * 100)

    # Load config and tokenizer

    cfg = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name_or_path)

    if use_test_document:
        # Use test document from user
        document = [
            ["Using Multiple Instance Learning to Build Multimodal Representations"],
            [
                "Abstract Image-text multimodal representation learning aligns data across modalities and enables important medical applications, e.g., image classification, visual grounding, and cross-modal retrieval.",
                "In this work, we establish a connection between multimodal representation learning and multiple instance learning.",
                "Based on this connection, we propose a generic framework for constructing permutation-invariant score functions with many existing multimodal representation learning approaches as special cases.",
                "Furthermore, we use the framework to derive a novel contrastive learning approach and demonstrate that our method achieves state-of-the-art results on a number of downstream tasks.",
            ],
            [
                "Introduction",
                "In this paper, we propose a framework for designing multimodal representation learning methods that encompasses previous approaches as special cases and implies a new algorithm for multimodal learning that advances the state of the art.",
                "Specifically, we establish a connection between self-supervised representation learning based on contrastive learning and multiple instance learning [3] and show that they",
            ],
        ]

        print("\n📚 Document structure:")
        print(f"   Section 1: {len(document[0])} sentence(s) (Title)")
        print(f"   Section 2: {len(document[1])} sentence(s) (Abstract)")
        print(f"   Section 3: {len(document[2])} sentence(s) (Introduction)")

        # Build hierarchy vectors
        input_ids, positions = build_coords_from_nested_list(
            document, tokenizer, max_length=cfg.data.max_seq_len, add_doc_token=True
        )

        # Extract hierarchy vectors
        level_0 = positions.level_enums[0].cpu()  # Token level [N]
        level_1 = positions.level_enums[1].cpu()  # Sentence level [N]
        level_2 = positions.level_enums[2].cpu()  # Section level [N]

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
    else:
        # Load from dataloader
        from lht.data.mlm import HierarchicalMLMDataModule

        dataset_info = (
            cfg.data.ds_info[0]
            if cfg.data.ds_info
            else {"path": "unknown", "split": "unknown"}
        )
        print(f"\n📚 Loading data from: {dataset_info.get('path', 'unknown')}")
        print(f"   Split: {dataset_info.get('split', 'unknown')}")
        data_module = HierarchicalMLMDataModule(cfg)
        data_module.setup("fit")

        # Get first batch
        train_loader = data_module.train_dataloader()
        first_batch = next(iter(train_loader))

        # Extract first example
        input_ids = first_batch["input_ids"][0]  # [N]
        positions_batch = first_batch[
            "positions"
        ]  # HierarchicalPositions with batch dim

        # Extract hierarchy vectors for first example
        level_0 = positions_batch.level_enums[0][0].cpu()  # Token level [N]
        level_1 = positions_batch.level_enums[1][0].cpu()  # Sentence level [N]
        level_2 = positions_batch.level_enums[2][0].cpu()  # Section level [N]

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu())

    # Calculate statistics
    seq_len = len(tokens)
    num_sentences = (level_1 > 0).sum().item()
    num_sections = (level_2 > 0).sum().item()

    print("\n📊 SEQUENCE STATISTICS")
    print(f"   Total tokens:        {seq_len}")
    print(f"   Sentence boundaries: {num_sentences}")
    print(f"   Section boundaries:  {num_sections}")

    # Print hierarchy vectors
    print(f"\n🔍 HIERARCHY VECTORS (first {min(max_tokens_display, seq_len)} tokens)")
    print("-" * 100)
    print(
        f"{'Pos':>4} | {'Token':>15} | {'L0 (Token)':>12} | {'L1 (Sent)':>12} | {'L2 (Sect)':>12} | {'Note'}"
    )
    print("-" * 100)

    display_limit = min(max_tokens_display, seq_len)

    for i in range(display_limit):
        token = tokens[i]
        l0_val = level_0[i].item()
        l1_val = level_1[i].item()
        l2_val = level_2[i].item()

        # Skip if showing boundaries only and this isn't a boundary
        if show_boundaries_only and l1_val == 0 and l2_val == 0:
            continue

        # Determine note
        note = ""
        if l2_val > 0:
            note = "🔴 Section boundary"
        elif l1_val > 0:
            note = "🟡 Sentence boundary"

        print(
            f"{i:4d} | {token:>15} | {l0_val:12d} | {l1_val:12d} | {l2_val:12d} | {note}"
        )

    if seq_len > max_tokens_display:
        print(f"... ({seq_len - max_tokens_display} more tokens)")

    print("-" * 100)

    # Explain attention pattern for a sample token
    print("\n💡 ATTENTION PATTERN EXAMPLE")
    print("-" * 100)

    # Find a sentence boundary token
    sent_boundary_idx = None
    for i in range(min(50, seq_len)):
        if level_1[i] > 0:
            sent_boundary_idx = i
            break

    if sent_boundary_idx is not None:
        print(
            f"\nFor token at position {sent_boundary_idx} ('{tokens[sent_boundary_idx]}'):"
        )
        print(f"  Level 0 enum: {level_0[sent_boundary_idx].item()}")
        print(
            f"  Level 1 enum: {level_1[sent_boundary_idx].item()} (sentence boundary)"
        )
        print(f"  Level 2 enum: {level_2[sent_boundary_idx].item()}")

        # Calculate which tokens it can attend to
        l0_enum = level_0[sent_boundary_idx].item()
        l1_enum = level_1[sent_boundary_idx].item()
        l2_enum = level_2[sent_boundary_idx].item()

        # Window sizes from config
        window_l0 = cfg.model.mlswa.window_size_per_level[0]
        window_l1 = cfg.model.mlswa.window_size_per_level[1]
        window_l2 = cfg.model.mlswa.window_size_per_level[2]

        print("\n  This token can attend to:")
        print(f"  • Level 0 (Token window={window_l0}):")
        print(
            f"    - Tokens with enum in range [{l0_enum - window_l0}, {l0_enum + window_l0}]"
        )

        if l1_enum > 0:
            print(f"  • Level 1 (Sentence window={window_l1}):")
            print(
                f"    - Sentence boundaries with enum in range [{l1_enum - window_l1}, {l1_enum + window_l1}]"
            )

        if l2_enum > 0:
            print(f"  • Level 2 (Section window={window_l2}):")
            print(
                f"    - Section boundaries with enum in range [{l2_enum - window_l2}, {l2_enum + window_l2}]"
            )

        # Find actual positions that match
        print("\n  Actual positions this token attends to (via cascading OR):")

        can_attend = torch.zeros(seq_len, dtype=torch.bool)

        # Level 0: check all positions
        l0_dist = torch.abs(level_0 - l0_enum)
        l0_match = l0_dist <= window_l0
        can_attend |= l0_match

        # Level 1: only check positions with non-zero L1
        if l1_enum > 0:
            l1_participates = level_1 > 0
            l1_dist = torch.abs(level_1 - l1_enum)
            l1_match = l1_participates & (l1_dist <= window_l1)
            can_attend |= l1_match

        # Level 2: only check positions with non-zero L2
        if l2_enum > 0:
            l2_participates = level_2 > 0
            l2_dist = torch.abs(level_2 - l2_enum)
            l2_match = l2_participates & (l2_dist <= window_l2)
            can_attend |= l2_match

        attend_positions = torch.where(can_attend)[0]
        print(f"    Total: {len(attend_positions)} positions out of {seq_len}")
        print(
            f"    Range: [{attend_positions[0].item()}, {attend_positions[-1].item()}]"
        )
        print(f"    Sparsity: {len(attend_positions) / seq_len * 100:.1f}%")

    # Print raw vectors for first N tokens (useful for debugging)
    print(f"\n📋 RAW VECTORS (first {min(20, seq_len)} tokens)")
    print("-" * 100)
    print(f"Level 0: {level_0[:20].tolist()}")
    print(f"Level 1: {level_1[:20].tolist()}")
    print(f"Level 2: {level_2[:20].tolist()}")

    # Show boundary statistics
    print("\n📈 BOUNDARY DISTRIBUTION")
    print("-" * 100)

    # Find all sentence boundary positions
    sent_boundaries = torch.where(level_1 > 0)[0]
    if len(sent_boundaries) > 1:
        sent_gaps = sent_boundaries[1:] - sent_boundaries[:-1]
        print(
            f"Sentence boundaries at positions: {sent_boundaries[:10].tolist()}"
            + (
                f" ... (+{len(sent_boundaries) - 10} more)"
                if len(sent_boundaries) > 10
                else ""
            )
        )
        print(
            f"  Mean gap between sentences: {sent_gaps.float().mean().item():.1f} tokens"
        )
        print(f"  Min gap: {sent_gaps.min().item()} tokens")
        print(f"  Max gap: {sent_gaps.max().item()} tokens")

    # Find all section boundary positions
    sect_boundaries = torch.where(level_2 > 0)[0]
    if len(sect_boundaries) > 1:
        sect_gaps = sect_boundaries[1:] - sect_boundaries[:-1]
        print(f"\nSection boundaries at positions: {sect_boundaries.tolist()}")
        print(
            f"  Mean gap between sections: {sect_gaps.float().mean().item():.1f} tokens"
        )
        print(f"  Min gap: {sect_gaps.min().item()} tokens")
        print(f"  Max gap: {sect_gaps.max().item()} tokens")

    print("\n" + "=" * 100)
    print("✅ Inspection complete!")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect hierarchical enumeration vectors"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_hierarchical.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to display",
    )
    parser.add_argument(
        "--boundaries-only",
        action="store_true",
        help="Only show boundary tokens",
    )
    parser.add_argument(
        "--from-dataloader",
        action="store_true",
        help="Load from dataloader instead of using test document",
    )
    args = parser.parse_args()

    visualize_hierarchy_vectors(
        args.config,
        max_tokens_display=args.max_tokens,
        show_boundaries_only=args.boundaries_only,
        use_test_document=not args.from_dataloader,
    )


if __name__ == "__main__":
    main()
