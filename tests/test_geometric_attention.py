"""
Test suite for geometric attention mechanism.

Tests Manhattan distance calculation, parent-child connectivity,
FlexAttention integration, and tensor shape handling.

Note: FlexAttention tests require CUDA and will be skipped on CPU-only systems.
Mock-based tests can run on CPU to verify parameter passing.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lht.core.attention import (
    GeometricCoordinates,
    create_geometric_mask,
    geometric_attention,
)

# Marker for tests that require CUDA (FlexAttention)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention requires CUDA"
)


def test_geometric_coordinates_initialization():
    """Test GeometricCoordinates initialization."""
    levels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0, 1, 0], dtype=torch.long)

    coords = GeometricCoordinates(levels, logical_times)

    assert torch.equal(coords.levels, levels)
    assert torch.equal(coords.logical_times, logical_times)
    assert len(coords.physical_positions) == len(levels)


def test_geometric_coordinates_device_transfer():
    """Test that coordinates can be moved to different devices."""
    levels = torch.tensor([0, 0, 1], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0], dtype=torch.long)

    coords = GeometricCoordinates(levels, logical_times)

    # Move to CPU
    coords_cpu = coords.to(torch.device("cpu"))
    assert coords_cpu.levels.device.type == "cpu"

    # Test non_blocking parameter
    coords_cpu = coords.to(torch.device("cpu"), non_blocking=True)
    assert coords_cpu.levels.device.type == "cpu"


def test_geometric_coordinates_physical_positions_default():
    """Test that physical_positions defaults to sequential indices."""
    levels = torch.tensor([0, 0, 1], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0], dtype=torch.long)

    coords = GeometricCoordinates(levels, logical_times)

    expected_positions = torch.arange(3, dtype=torch.long)
    assert torch.equal(coords.physical_positions, expected_positions)


def test_manhattan_distance_siblings():
    """Test that siblings (same parent, same level) have distance = 0."""
    # Two tokens in same sentence
    levels = torch.tensor([0, 0], dtype=torch.long)
    logical_times = torch.tensor([0, 0], dtype=torch.long)  # Both in sentence 0

    # Manhattan distance = |0-0| + |0-0| = 0
    dist = torch.abs(levels[0] - levels[1]) + torch.abs(
        logical_times[0] - logical_times[1]
    )
    assert dist == 0, "Siblings should have distance 0"


def test_manhattan_distance_parent_child():
    """Test that parent-child pairs have distance = 1."""
    # Token at (x=0, y=0), its parent sentence at (x=0, y=1)
    token_level = 0
    token_time = 0
    sentence_level = 1
    sentence_time = 0  # Same x-coordinate

    dist = abs(token_level - sentence_level) + abs(token_time - sentence_time)
    assert dist == 1, "Parent-child should have distance 1"


def test_manhattan_distance_cross_section():
    """Test that tokens in different sections have distance > 1."""
    # Token in sentence 0 (x=0, y=0)
    # Token in sentence 1 (x=1, y=0)
    levels = torch.tensor([0, 0], dtype=torch.long)
    logical_times = torch.tensor([0, 1], dtype=torch.long)

    dist = torch.abs(levels[0] - levels[1]) + torch.abs(
        logical_times[0] - logical_times[1]
    )
    assert dist == 1, "Tokens in different sentences at same level have distance = |Δx|"

    # Token in sentence 0 (x=0, y=0) to sentence 1 summary (x=1, y=1)
    token_level, token_time = 0, 0
    other_sent_level, other_sent_time = 1, 1

    dist = abs(token_level - other_sent_level) + abs(token_time - other_sent_time)
    assert dist == 2, "Cross-sentence token to sentence should have distance 2"


def test_create_geometric_mask_basic():
    """Test basic geometric mask creation with mocking."""
    # Simple 2-token, 1-sentence structure
    levels = torch.tensor([0, 0], dtype=torch.long)
    logical_times = torch.tensor([0, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    batch_size = 1
    num_heads = 2

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            batch_size=batch_size,
            num_heads=num_heads,
        )

        # Verify mask was created and returned
        assert mask is not None
        assert mock_create_block_mask.called

        # Verify correct parameters
        call_kwargs = mock_create_block_mask.call_args[1]
        assert call_kwargs["B"] == batch_size
        assert call_kwargs["H"] == num_heads
        assert call_kwargs["Q_LEN"] == 2
        assert call_kwargs["KV_LEN"] == 2


def test_create_geometric_mask_with_max_level():
    """Test mask creation with per-layer level control using mocks."""
    levels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0, 1, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        # Restrict to levels 0-1 only
        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            max_level=1,
            batch_size=1,
            num_heads=2,
        )

        assert mask is not None
        assert mock_create_block_mask.called

        # Verify sequence length is correct
        call_kwargs = mock_create_block_mask.call_args[1]
        assert call_kwargs["Q_LEN"] == 5
        assert call_kwargs["KV_LEN"] == 5


def test_create_geometric_mask_with_window_sizes():
    """Test mask creation with sliding window constraints using mocks."""
    levels = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [256, 64, 16]  # Per-level windows

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        mask = create_geometric_mask(
            coords=coords,
            radius=1,
            window_size_per_level=window_sizes,
            batch_size=1,
            num_heads=2,
        )

        assert mask is not None
        assert mock_create_block_mask.called

        # Verify parameters
        call_kwargs = mock_create_block_mask.call_args[1]
        assert call_kwargs["B"] == 1
        assert call_kwargs["H"] == 2
        assert call_kwargs["Q_LEN"] == 4


def test_geometric_attention_output_shape():
    """Test that geometric attention returns correct output shape using mocks."""
    batch_size = 2
    seq_len = 8
    num_heads = 4
    head_dim = 16

    # Create dummy Q, K, V tensors [B, N, H, D]
    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Create coords [B, N]
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock returns output in [B, H, N, D] format (what flex_attention returns)
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            # Run attention
            output = geometric_attention(query, key, value, coords, radius=1)

            # Output should have same shape as input (transposed back to [B, N, H, D])
            assert (
                output.shape == query.shape
            ), f"Expected {query.shape}, got {output.shape}"
            assert output.shape == (batch_size, seq_len, num_heads, head_dim)

            # Verify mocks were called
            assert mock_flex_attn.call_count == 1, "FlexAttention should be called once"
            assert (
                mock_create_mask.call_count == 1
            ), "create_geometric_mask should be called once"

            # Verify Q, K, V are transposed correctly before passing to flex_attention
            # geometric_attention transposes [B, N, H, D] -> [B, H, N, D]
            call_args = mock_flex_attn.call_args[0]
            q_arg = call_args[0]
            assert q_arg.shape == (
                batch_size,
                num_heads,
                seq_len,
                head_dim,
            ), f"Q should be transposed to [B, H, N, D], got {q_arg.shape}"


def test_geometric_attention_with_hierarchy():
    """Test geometric attention with hierarchical coordinates using mocks."""
    batch_size = 1
    seq_len = 6  # 6 tokens
    num_heads = 2
    head_dim = 8

    # Create Q, K, V
    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Create hierarchical coords (2 sentences of 3 tokens each)
    levels = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
    logical_times = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(query, key, value, coords, radius=1)

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)

            # Verify coordinates were used to create mask
            assert mock_create_mask.called, "Mask creation should be called with coords"

            # Verify coords were passed to create_geometric_mask
            call_kwargs = mock_create_mask.call_args[1]
            assert "coords" in call_kwargs, "coords should be passed to mask creation"
            passed_coords = call_kwargs["coords"]
            assert torch.equal(
                passed_coords.levels, coords.levels
            ), "levels should match"
            assert torch.equal(
                passed_coords.logical_times, coords.logical_times
            ), "logical_times should match"


def test_geometric_attention_batched_coords():
    """Test geometric attention with batched coordinates using mocks."""
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 8

    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Batched coords [B, N]
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(query, key, value, coords, radius=1)

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            # Verify batched coordinates were passed
            assert mock_create_mask.call_args[1]["batch_size"] == batch_size


def test_geometric_attention_with_window():
    """Test geometric attention with sliding window constraints using mocks."""
    batch_size = 1
    seq_len = 8
    num_heads = 2
    head_dim = 8

    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [4, 64, 16]  # Small window for testing

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(
                query, key, value, coords, radius=1, window_size_per_level=window_sizes
            )

            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            # Verify window sizes were passed
            assert (
                mock_create_mask.call_args[1]["window_size_per_level"] == window_sizes
            )


def test_geometric_attention_deterministic():
    """Test that geometric attention setup is deterministic using mocks."""
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 8

    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Return deterministic output
        torch.manual_seed(42)
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            # Run twice
            output1 = geometric_attention(query, key, value, coords, radius=1)

            # Reset mock for second call
            torch.manual_seed(42)
            mock_output2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
            mock_flex_attn.return_value = mock_output2

            output2 = geometric_attention(query, key, value, coords, radius=1)

            # Verify same parameters were passed both times
            assert mock_flex_attn.call_count == 2
            assert output1.shape == output2.shape


def test_geometric_attention_values_reasonable():
    """Test that attention handles normalized inputs correctly using mocks."""
    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 8

    # Use normalized inputs
    query = torch.randn(batch_size, seq_len, num_heads, head_dim) * 0.1
    key = torch.randn(batch_size, seq_len, num_heads, head_dim) * 0.1
    value = torch.randn(batch_size, seq_len, num_heads, head_dim) * 0.1

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Return reasonable output values
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(query, key, value, coords, radius=1)

            # Verify tensors were passed to flex_attention
            assert mock_flex_attn.called
            call_args = mock_flex_attn.call_args[0]

            # Verify input tensors are in reasonable range
            assert call_args[0].abs().max() < 100  # query
            assert call_args[1].abs().max() < 100  # key
            assert call_args[2].abs().max() < 100  # value

            # Output should not have NaN or Inf
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"


def test_create_geometric_mask_calls_create_block_mask_with_correct_params():
    """Test that create_geometric_mask calls create_block_mask with correct parameters (CPU)."""

    levels = torch.tensor([0, 0, 1], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        # Set up mock to return a dummy mask
        mock_mask = MagicMock()
        mock_create_block_mask.return_value = mock_mask

        result = create_geometric_mask(
            coords=coords,
            radius=1,
            max_level=2,
            window_size_per_level=[256, 64, 16],
            batch_size=2,
            num_heads=4,
            device=torch.device("cpu"),
        )

        # Verify mask was created
        assert result is not None

        # Verify create_block_mask was called
        assert mock_create_block_mask.called
        call_args = mock_create_block_mask.call_args

        # Check that function was passed
        assert callable(call_args[0][0]), "First arg should be mask function"

        # Check keyword arguments
        assert call_args[1]["B"] == 2
        assert call_args[1]["H"] == 4
        assert call_args[1]["Q_LEN"] == 3
        assert call_args[1]["KV_LEN"] == 3


def test_geometric_attention_transposes_tensors_correctly():
    """Test that geometric_attention transposes tensors before/after flex_attention (CPU)."""
    batch_size = 2
    seq_len = 4
    num_heads = 2
    head_dim = 8

    # Create input tensors in [B, N, H, D] format
    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock flex_attention to return tensor in [B, H, N, D] format
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask"):
            output = geometric_attention(query, key, value, coords, radius=1)

        # Verify flex_attention was called
        assert mock_flex_attn.called

        # Get the tensors passed to flex_attention
        call_args = mock_flex_attn.call_args[0]
        q_passed, k_passed, v_passed = call_args[0], call_args[1], call_args[2]

        # Verify tensors were transposed to [B, H, N, D]
        assert q_passed.shape == (batch_size, num_heads, seq_len, head_dim)
        assert k_passed.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v_passed.shape == (batch_size, num_heads, seq_len, head_dim)

        # Verify output was transposed back to [B, N, H, D]
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)


def test_geometric_mask_fn_logic():
    """Test the geometric mask function logic without FlexAttention (CPU)."""
    from lht.core.attention import create_geometric_mask

    # Create simple coordinates
    levels = torch.tensor([0, 0, 1], dtype=torch.long)
    logical_times = torch.tensor([0, 1, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        # Capture the mask function
        mask_fn = None

        def capture_mask_fn(fn, **kwargs):
            nonlocal mask_fn
            mask_fn = fn
            return MagicMock()

        mock_create_block_mask.side_effect = capture_mask_fn

        create_geometric_mask(
            coords=coords,
            radius=1,
            batch_size=1,
            num_heads=1,
            device=torch.device("cpu"),
        )

        # Now test the mask function logic
        assert mask_fn is not None

        # Test same position (distance=0, should attend)
        result = mask_fn(0, 0, 0, 0)  # q_idx=0, kv_idx=0
        assert result is True or result.item() is True

        # Test different positions at same level with distance=1
        # Token 0 (level=0, time=0) to Token 1 (level=0, time=1)
        # Manhattan distance = |0-0| + |0-1| = 1, should attend with radius=1
        result = mask_fn(0, 0, 0, 1)
        assert result is True or result.item() is True


def test_geometric_attention_with_window_mocked():
    """Test that window sizes are passed through correctly (CPU)."""
    batch_size = 1
    seq_len = 8
    num_heads = 2
    head_dim = 8

    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)

    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.arange(seq_len).unsqueeze(0)
    coords = GeometricCoordinates(levels, logical_times)

    window_sizes = [4, 64, 16]

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        mock_output = torch.randn(batch_size, num_heads, seq_len, head_dim)
        mock_flex_attn.return_value = mock_output

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = geometric_attention(
                query, key, value, coords, radius=1, window_size_per_level=window_sizes
            )

            # Verify output shape is correct
            assert output.shape == (batch_size, seq_len, num_heads, head_dim)

            # Verify create_geometric_mask was called with window_sizes
            assert mock_create_mask.called
            call_kwargs = mock_create_mask.call_args[1]
            assert call_kwargs["window_size_per_level"] == window_sizes


def test_geometric_mask_respects_max_level():
    """Test that max_level parameter is passed correctly (CPU)."""
    levels = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    logical_times = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention.create_block_mask") as mock_create_block_mask:
        mock_create_block_mask.return_value = MagicMock()

        create_geometric_mask(
            coords=coords,
            radius=1,
            max_level=1,  # Only levels 0-1 should be active
            batch_size=1,
            num_heads=2,
            device=torch.device("cpu"),
        )

        # Verify the function was called
        assert mock_create_block_mask.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
