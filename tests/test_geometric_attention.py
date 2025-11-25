"""
Test suite for Multi-Level Sliding Window Attention (ML-SWA) mechanism.

Tests hierarchical position enumeration, cascading window masks,
FlexAttention integration, and tensor shape handling.

Note: FlexAttention tests require CUDA and will be skipped on CPU-only systems.
Mock-based tests can run on CPU to verify parameter passing.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lht.core.attention import (
    HierarchicalPositions,
    create_hierarchical_mask,
    mlswa_attention,
)

# Marker for tests that require CUDA (FlexAttention)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention requires CUDA"
)


def test_hierarchical_positions_initialization():
    """Test HierarchicalPositions initialization."""
    level_0 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 0, 2, 0], dtype=torch.long)
    level_2 = torch.tensor([1, 0, 0, 0, 2], dtype=torch.long)

    positions = HierarchicalPositions([level_0, level_1, level_2])

    assert positions.num_levels == 3
    assert torch.equal(positions.level_enums[0], level_0)
    assert torch.equal(positions.level_enums[1], level_1)
    assert torch.equal(positions.level_enums[2], level_2)


def test_hierarchical_positions_device_transfer():
    """Test that positions can be moved to different devices."""
    level_0 = torch.tensor([1, 2, 3], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2], dtype=torch.long)

    positions = HierarchicalPositions([level_0, level_1])

    # Move to CPU
    positions_cpu = positions.to(torch.device("cpu"))
    assert positions_cpu.level_enums[0].device.type == "cpu"
    assert positions_cpu.level_enums[1].device.type == "cpu"


def test_sparse_enumeration_semantics():
    """Test that zero means non-participant in higher levels."""
    # 5 tokens, 2 sentence boundaries (at positions 2 and 4), 1 section boundary (at position 4)
    level_0 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)  # all participate
    level_1 = torch.tensor(
        [0, 0, 1, 0, 2], dtype=torch.long
    )  # only pos 2 and 4 participate
    level_2 = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)  # only pos 4 participates

    # Level 0: all tokens participate
    assert torch.all(level_0 > 0)

    # Level 1: only boundary positions participate
    assert (level_1 > 0).sum() == 2

    # Level 2: only section boundary participates
    assert (level_2 > 0).sum() == 1


def test_create_hierarchical_mask_basic():
    """Test basic hierarchical mask creation with mocking."""
    level_0 = torch.tensor([1, 2, 3], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    batch_size = 1
    num_heads = 2
    window_sizes = [2, 1]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        mask = create_hierarchical_mask(
            positions,
            window_size_per_level=window_sizes,
            batch_size=batch_size,
            num_heads=num_heads,
        )

        # Verify create_block_mask was called once
        assert mock_create_block_mask.call_count == 1

        # Verify parameters
        call_args = mock_create_block_mask.call_args
        assert call_args.kwargs["B"] == batch_size
        assert call_args.kwargs["H"] == num_heads
        assert call_args.kwargs["Q_LEN"] == 3
        assert call_args.kwargs["KV_LEN"] == 3

        # Verify mask function was passed
        assert callable(call_args.args[0])


def test_create_hierarchical_mask_with_max_level():
    """Test mask creation with max_level parameter."""
    level_0 = torch.tensor([1, 2, 3], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2], dtype=torch.long)
    level_2 = torch.tensor([1, 0, 0], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1, level_2])

    window_sizes = [2, 1, 1]

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        # Only use first 2 levels
        create_hierarchical_mask(
            positions,
            window_size_per_level=window_sizes,
            max_level=1,  # Only levels 0 and 1
            batch_size=1,
            num_heads=1,
        )

        assert mock_create_block_mask.call_count == 1


def test_mlswa_attention_mock():
    """Test mlswa_attention function with mocking."""
    B, N, H, D = 2, 4, 2, 8
    query = torch.randn(B, N, H, D)
    key = torch.randn(B, N, H, D)
    value = torch.randn(B, N, H, D)

    level_0 = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2, 0], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [2, 1]

    with patch("lht.core.attention.create_hierarchical_mask") as mock_create_mask:
        with patch("lht.core.attention._compiled_mlswa_attention") as mock_flex_attn:
            mock_mask = MagicMock()
            mock_create_mask.return_value = mock_mask
            mock_flex_attn.return_value = torch.randn(B, H, N, D)

            output = mlswa_attention(
                query, key, value, positions, window_size_per_level=window_sizes
            )

            # Verify mask creation was called
            assert mock_create_mask.call_count == 1

            # Verify flex_attention was called
            assert mock_flex_attn.call_count == 1

            # Verify output shape
            assert output.shape == (B, N, H, D)


def test_mlswa_attention_shape():
    """Test that mlswa_attention handles tensor shapes correctly."""
    B, N, H, D = 1, 4, 2, 8
    query = torch.randn(B, N, H, D)
    key = torch.randn(B, N, H, D)
    value = torch.randn(B, N, H, D)

    level_0 = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 2, 0], dtype=torch.long)
    positions = HierarchicalPositions([level_0, level_1])

    window_sizes = [2, 1]

    with patch("lht.core.attention.create_hierarchical_mask") as mock_create_mask:
        mock_create_mask.return_value = MagicMock()

        with patch("lht.core.attention._compiled_mlswa_attention") as mock_flex_attn:
            # Mock returns correct shape
            mock_flex_attn.return_value = torch.randn(B, H, N, D)

            output = mlswa_attention(
                query, key, value, positions, window_size_per_level=window_sizes
            )

            # Check the flex_attention was called with transposed tensors [B, H, N, D]
            call_args = mock_flex_attn.call_args.args
            assert call_args[0].shape == (B, H, N, D)  # query transposed
            assert call_args[1].shape == (B, H, N, D)  # key transposed
            assert call_args[2].shape == (B, H, N, D)  # value transposed

            # Check output is transposed back to [B, N, H, D]
            assert output.shape == (B, N, H, D)


def test_batched_hierarchical_positions():
    """Test HierarchicalPositions with batched tensors [B, N]."""
    B, N = 2, 5
    level_0 = torch.randint(1, 10, (B, N), dtype=torch.long)
    level_1 = torch.randint(0, 3, (B, N), dtype=torch.long)

    positions = HierarchicalPositions([level_0, level_1])

    assert positions.num_levels == 2
    assert positions.level_enums[0].shape == (B, N)
    assert positions.level_enums[1].shape == (B, N)


def test_cascading_window_logic():
    """Test that cascading windows work as expected (conceptual test)."""
    # Conceptual: All positions attend via level 0
    # Boundary positions (non-zero in level 1) add level 1 connectivity
    # This is verified by the mask function logic, not directly testable without CUDA

    level_0 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    level_1 = torch.tensor([1, 0, 0, 2, 0], dtype=torch.long)  # boundaries at 0, 3

    # Level 0: all positions can attend to each other (within window)
    # Level 1: positions 0 and 3 can additionally attend to each other (if within window)

    # This is the expected behavior - we verify the mask function exists
    positions = HierarchicalPositions([level_0, level_1])

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        create_hierarchical_mask(
            positions,
            window_size_per_level=[3, 2],
            batch_size=1,
            num_heads=1,
        )

        # The mask function should be created successfully
        assert mock_create_block_mask.called


@requires_cuda
def test_mlswa_attention_cuda():
    """Test mlswa_attention on CUDA if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, N, H, D = 1, 8, 2, 4
    device = torch.device("cuda")

    query = torch.randn(B, N, H, D, device=device)
    key = torch.randn(B, N, H, D, device=device)
    value = torch.randn(B, N, H, D, device=device)

    level_0 = torch.arange(1, N + 1, dtype=torch.long, device=device)
    level_1 = torch.zeros(N, dtype=torch.long, device=device)
    level_1[3] = 1
    level_1[7] = 2

    positions = HierarchicalPositions([level_0, level_1])
    window_sizes = [4, 2]

    output = mlswa_attention(
        query, key, value, positions, window_size_per_level=window_sizes
    )

    assert output.shape == (B, N, H, D)
    assert output.device.type == "cuda"
    assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
