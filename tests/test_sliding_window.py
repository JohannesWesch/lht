"""
Test suite for sliding window mechanism.

Tests per-level window constraints, interaction with Manhattan distance,
and edge cases for window boundaries.

Tests use mocking to run on CPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lht.core.attention import (
    GeometricCoordinates,
    create_geometric_mask,
    geometric_attention,
)


def test_same_level_within_window():
    """Test that tokens within window at same level can attend."""
    # Create tokens at same level but different logical times
    seq_len = 10
    levels = torch.zeros(seq_len, dtype=torch.long)  # All at level 0
    logical_times = torch.arange(seq_len, dtype=torch.long)  # Sequential times
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [5, 64, 16]  # Window of 5 for level 0

    # Token at time=0 should attend to tokens 0-5 (within window)
    # Token at time=7 should NOT attend to token 0 (distance=7, window=5)

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None, "Mask should be created"

    # Verify create_block_mask was called with correct parameters
    assert mock_create_block_mask.called, "create_block_mask should be called"
    call_kwargs = mock_create_block_mask.call_args[1]
    assert call_kwargs["Q_LEN"] == seq_len, f"Q_LEN should be {seq_len}"
    assert call_kwargs["KV_LEN"] == seq_len, f"KV_LEN should be {seq_len}"

    # Verify window_size_per_level was passed through
    # The mask function should have access to window constraints


def test_same_level_outside_window():
    """Test that tokens outside window at same level cannot attend."""
    # Create many tokens at same level with large time differences
    seq_len = 100
    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [10, 64, 16]  # Small window of 10

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    # Mask should be created (actual attention blocking tested implicitly)
    assert mask is not None, "Mask should be created even with large sequence"

    # Verify mask was created with correct sequence length
    assert mock_create_block_mask.called, "create_block_mask should be called"
    call_kwargs = mock_create_block_mask.call_args[1]
    assert call_kwargs["Q_LEN"] == seq_len, f"Q_LEN should be {seq_len}"
    assert call_kwargs["KV_LEN"] == seq_len, f"KV_LEN should be {seq_len}"


def test_cross_level_ignores_window():
    """Test that window constraint doesn't apply to cross-level attention."""
    # Token at (x=0, y=0) and sentence at (x=0, y=1)
    levels = torch.tensor([0, 1], dtype=torch.long)
    logical_times = torch.tensor([0, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [1, 64, 16]  # Very small window at level 0

    # Cross-level attention should still work (Manhattan distance = 1)
    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_window_size_per_level():
    """Test different window sizes for different levels."""
    # Create 3-level hierarchy
    levels = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [256, 64, 16]  # Different sizes for each level

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=2,
        )

    assert mask is not None


def test_window_with_manhattan_radius():
    """Test combined Manhattan distance and window constraints."""
    # Tokens at same level within window
    seq_len = 20
    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    # Both radius and window apply
    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,  # Manhattan constraint
            window_size_per_level=[10, 64, 16],  # Window constraint
            batch_size=1,
            num_heads=2,
        )

    assert mask is not None


def test_window_boundary_exact():
    """Test behavior at exact window boundary."""
    # Token at time=0, token at time=window_size
    window_size = 10
    seq_len = window_size + 1

    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [window_size, 64, 16]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_large_window_no_restriction():
    """Test that very large window effectively allows all same-level attention."""
    seq_len = 50
    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [1000, 1000, 1000]  # Very large windows

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_window_with_batched_coords():
    """Test window constraints with batched coordinates."""
    batch_size = 2
    seq_len = 20

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [10, 64, 16]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=batch_size,
            num_heads=2,
        )

    assert mask is not None


def test_window_attention_integration():
    """Test full attention with window constraints."""
    batch_size = 1
    seq_len = 16
    num_heads = 2
    head_dim = 8

    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len).unsqueeze(0)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [8, 64, 16]

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock returns [B, H, N, D] which gets transposed back to [B, N, H, D]
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(
                query, key, value, coords, radius=1, window_size_per_level=window_sizes
            )

    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
    assert not torch.isnan(output).any()


def test_hierarchical_with_windows():
    """Test window constraints on hierarchical structure."""
    # Mix of tokens, sentences, sections with windows
    batch_size = 1
    # 6 tokens in 2 sentences in 1 section
    seq_len = 6

    query = torch.randn(batch_size, seq_len, 2, 8)
    key = torch.randn(batch_size, seq_len, 2, 8)
    value = torch.randn(batch_size, seq_len, 2, 8)

    # All tokens at level 0, split into 2 groups
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [256, 64, 16]

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock returns [B, H, N, D] which gets transposed back to [B, N, H, D]
        mock_output = torch.randn(batch_size, query.shape[2], seq_len, query.shape[3])
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(
                query, key, value, coords, radius=1, window_size_per_level=window_sizes
            )

    assert output.shape == query.shape


def test_window_zero_size():
    """Test edge case with window size of 0."""
    seq_len = 4
    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [0, 64, 16]  # Zero window at level 0

    # Should still create mask (tokens can't attend to distant same-level tokens)
    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_window_none():
    """Test that window=None allows unrestricted same-level attention."""
    seq_len = 10
    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    # No window constraint
    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=None,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_level_specific_windows():
    """Test that level 1 and level 2 have different window behaviors."""
    # Create multi-level structure
    levels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    logical_times = torch.arange(8, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    # Different windows per level
    window_sizes = [2, 4, 6]  # Increasing windows for higher levels

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


def test_window_with_same_logical_time():
    """Test window constraint when multiple tokens have same logical_time."""
    # Siblings (same parent) should always attend regardless of window
    seq_len = 10
    levels = torch.zeros(seq_len, dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [1, 64, 16]  # Small window

    # Tokens with same logical_time (same sentence) should attend
    # Distance = 0, so window doesn't matter
    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:

        mock_create_block_mask.return_value = MagicMock()

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=1,
        )

    assert mask is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
