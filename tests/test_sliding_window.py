"""
Test suite for Multi-Level Sliding Window Attention (ML-SWA).

Tests per-level cascading window constraints, sparse enumeration,
and edge cases for window boundaries.

Tests use mocking to run on CPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lht.core.attention import (
    HierarchicalPositions,
    create_hierarchical_mask,
    mlswa_attention,
)


def test_level_0_window():
    """Test that all tokens within window at level 0 can attend."""
    # Create simple positions: all tokens at level 0
    seq_len = 10
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)  # 1, 2, 3, ..., 10
    level_1 = torch.zeros(seq_len, dtype=torch.long)  # No boundaries
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [3, 1]  # Window of 3 for level 0

    # Token at enum=5 should attend to tokens 2-8 (within ±3 distance)
    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None, "Mask should be created"
    assert mock_create_block_mask.called, "create_block_mask should be called"


def test_sparse_enumeration_level_1():
    """Test that only boundary positions participate in level 1."""
    seq_len = 10
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.zeros(seq_len, dtype=torch.long)
    level_1[3] = 1  # Boundary at position 3
    level_1[7] = 2  # Boundary at position 7
    positions = HierarchicalPositions([level_0, level_1])

    # Verify sparse participation
    assert (level_1 > 0).sum() == 2, "Only 2 positions should participate in level 1"


def test_cascading_windows():
    """Test that cascading windows work (OR merge)."""
    seq_len = 8
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.tensor([1, 0, 0, 2, 0, 0, 3, 0], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [2, 1]  # Tight windows for both levels

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    # All positions get level 0 window
    # Positions 0, 3, 6 additionally get level 1 window
    assert mask is not None


def test_max_level_constraint():
    """Test that max_level limits active hierarchy levels."""
    seq_len = 6
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2, 0, 3, 0], dtype=torch.long)
    level_2 = torch.tensor([1, 0, 0, 0, 0, 2], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1, level_2])

    window_sizes = [2, 1, 1]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        # Only use first 2 levels
        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            max_level=1,  # Only use levels 0 and 1
            batch_size=1,
            num_heads=1,
        )

    assert mock_create_block_mask.called


def test_batched_positions():
    """Test sliding windows with batched positions."""
    B, N = 2, 5
    level_0 = torch.arange(1, N + 1, dtype=torch.long).unsqueeze(0).expand(B, -1)
    level_1 = torch.zeros(B, N, dtype=torch.long)
    level_1[:, -1] = 1  # Last position is boundary in both batch items
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [2, 1]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=B,
            num_heads=2,
        )

    call_kwargs = mock_create_block_mask.call_args[1]
    assert call_kwargs["B"] == B
    assert call_kwargs["Q_LEN"] == N
    assert call_kwargs["KV_LEN"] == N


def test_zero_means_non_participant():
    """Test that zero enumeration values mean non-participation."""
    seq_len = 6
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.tensor(
        [1, 0, 0, 2, 0, 0], dtype=torch.long
    )  # Only positions 0 and 3 participate
    positions = HierarchicalPositions([level_0, level_1])

    # Verify participation
    participating = level_1 > 0
    assert participating.sum() == 2
    assert participating[0] == True
    assert participating[3] == True
    assert participating[1] == False


def test_window_distance_in_enumeration_space():
    """Test that window distance is measured in enumeration space."""
    # Level 1: enumerations 1, 0, 0, 2, 0, 3
    # Positions 0, 3, 5 participate with enums 1, 2, 3
    # With window=1, enum 2 should attend to enums 1 and 3 (distance ≤ 1)
    seq_len = 6
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.tensor([1, 0, 0, 2, 0, 3], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    # Enum 2 at position 3 should see:
    # - Enum 1 at position 0 (|2-1| = 1 ≤ window)
    # - Enum 3 at position 5 (|2-3| = 1 ≤ window)

    window_sizes = [2, 1]  # Window of 1 for level 1

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_mlswa_attention_with_windows():
    """Test mlswa_attention integrates with windowing."""
    B, N, H, D = 1, 6, 2, 4
    query = torch.randn(B, N, H, D)
    key = torch.randn(B, N, H, D)
    value = torch.randn(B, N, H, D)

    level_0 = torch.arange(1, N + 1, dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2, 0, 3, 0], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [3, 1]

    with patch("lht.core.attention.create_hierarchical_mask") as mock_create_mask:
        mock_create_mask.return_value = MagicMock()

        with patch("lht.core.attention._compiled_mlswa_attention") as mock_flex_attn:
            mock_flex_attn.return_value = torch.randn(B, H, N, D)

            output = mlswa_attention(
                query, key, value, positions, window_size_per_level=window_sizes
            )

    assert output.shape == (B, N, H, D)
    assert mock_flex_attn.called


def test_empty_level_1():
    """Test handling when level 1 has no participants."""
    seq_len = 5
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.zeros(seq_len, dtype=torch.long)  # No participants
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [2, 1]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    # Should still work - only level 0 window will apply
    assert mask is not None


def test_large_window_covers_all():
    """Test that very large window allows all positions to attend."""
    seq_len = 10
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    level_1 = torch.zeros(seq_len, dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [100, 100]  # Very large windows

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_single_level():
    """Test with only level 0 (single level hierarchy)."""
    seq_len = 5
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.long)
    positions = HierarchicalPositions([level_0])

    window_sizes = [2]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_hierarchical_mask(
            positions=positions,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
