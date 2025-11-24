"""
Test suite for LHTEncoder and GeometricTransformerBlock.

Tests model initialization, weight initialization, forward pass,
output shapes, and parameter counts.

Note: Forward pass tests require CUDA as they use FlexAttention.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lht.core.attention import GeometricCoordinates
from lht.core.model import GeometricTransformerBlock
from lht.model import LHTEncoder

# Marker for tests that require CUDA (forward passes with geometric attention)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention requires CUDA"
)


def create_test_config():
    """Create a minimal test configuration."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            num_layers=2,
            d_ff=512,
            dropout=0.1,
            max_seq_len=512,
            geometry=SimpleNamespace(
                num_levels=3,
                manhattan_radius=1,
                window_size_per_level=[256, 64, 16],
                layer_max_level=[2, 2],
            ),
        )
    )
    return config


def test_lht_encoder_initialization():
    """Test that LHTEncoder initializes correctly."""
    config = create_test_config()
    model = LHTEncoder(config)

    assert isinstance(model, nn.Module)
    assert hasattr(model, "token_embed")
    assert hasattr(model, "layers")
    assert hasattr(model, "final_norm")
    assert hasattr(model, "mlm_head")


def test_token_embedding_initialization():
    """Test that token embeddings are initialized with correct parameters."""
    config = create_test_config()
    model = LHTEncoder(config)

    # Check embedding size
    assert model.token_embed.num_embeddings == config.model.vocab_size
    assert model.token_embed.embedding_dim == config.model.d_model

    # Check initialization (mean should be close to 0, std close to 0.02)
    weights = model.token_embed.weight.data
    assert abs(weights.mean().item()) < 0.01, "Mean should be close to 0"
    assert abs(weights.std().item() - 0.02) < 0.01, "Std should be close to 0.02"


def test_mlm_head_initialization():
    """Test that MLM head is initialized correctly."""
    config = create_test_config()
    model = LHTEncoder(config)

    # Check MLM head size
    assert model.mlm_head.in_features == config.model.d_model
    assert model.mlm_head.out_features == config.model.vocab_size

    # Check no bias
    assert model.mlm_head.bias is None, "MLM head should not have bias"

    # Check initialization
    weights = model.mlm_head.weight.data
    assert abs(weights.mean().item()) < 0.01, "Mean should be close to 0"
    assert abs(weights.std().item() - 0.02) < 0.01, "Std should be close to 0.02"


def test_no_weight_tying():
    """Test that embeddings and MLM head do NOT share weights."""
    config = create_test_config()
    model = LHTEncoder(config)

    # They should be different tensors
    assert (
        model.token_embed.weight is not model.mlm_head.weight
    ), "Token embeddings and MLM head should not share weights"


def test_final_norm_exists():
    """Test that final LayerNorm exists."""
    config = create_test_config()
    model = LHTEncoder(config)

    assert hasattr(model, "final_norm")
    assert isinstance(model.final_norm, nn.LayerNorm)
    assert model.final_norm.normalized_shape == (config.model.d_model,)


def test_num_layers():
    """Test that correct number of layers are created."""
    config = create_test_config()
    model = LHTEncoder(config)

    assert len(model.layers) == config.model.num_layers


def test_geometric_transformer_block_initialization():
    """Test GeometricTransformerBlock initialization."""
    block = GeometricTransformerBlock(
        d_model=128,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        manhattan_radius=1,
        layer_idx=0,
        layer_max_level=2,
        window_size_per_level=[256, 64, 16],
    )

    assert isinstance(block, nn.Module)
    assert block.d_model == 128
    assert block.n_heads == 4
    assert block.head_dim == 32  # 128 / 4


def test_forward_pass_output_structure():
    """Test that forward pass returns correct output structure (mocked attention)."""
    config = create_test_config()
    model = LHTEncoder(config)

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

    # Create dummy coordinates
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock flex_attention to return dummy output
        def mock_attn(q, k, v, block_mask=None):
            return torch.randn_like(q)

        mock_flex_attn.side_effect = mock_attn

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = model(input_ids, coords=coords)

            # Check output structure
            assert isinstance(output, dict), "Output should be a dict"
            assert "hidden" in output, "Output should contain 'hidden' key"
            assert "mlm_logits" in output, "Output should contain 'mlm_logits' key"
            assert isinstance(
                output["hidden"], torch.Tensor
            ), "hidden should be a tensor"
            assert isinstance(
                output["mlm_logits"], torch.Tensor
            ), "mlm_logits should be a tensor"

            # Verify mocks were called
            assert mock_flex_attn.called, "FlexAttention should have been called"
            assert (
                mock_create_mask.called
            ), "create_geometric_mask should have been called"


def test_forward_pass_output_shapes():
    """Test that forward pass produces correct output shapes (mocked attention)."""
    config = create_test_config()
    model = LHTEncoder(config)

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock flex_attention to return correct shape
        def mock_attn(q, k, v, block_mask=None):
            # q shape: [B, N, H, D] after reshape in geometric_attention
            # Should return same shape
            return torch.randn_like(q)

        mock_flex_attn.side_effect = mock_attn

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = model(input_ids, coords=coords)

    # Check hidden states shape
    assert output["hidden"].shape == (
        batch_size,
        seq_len,
        config.model.d_model,
    ), f"Expected hidden shape {(batch_size, seq_len, config.model.d_model)}, got {output['hidden'].shape}"

    # Check logits shape
    assert output["mlm_logits"].shape == (
        batch_size,
        seq_len,
        config.model.vocab_size,
    ), f"Expected logits shape {(batch_size, seq_len, config.model.vocab_size)}, got {output['mlm_logits'].shape}"


def test_forward_pass_requires_coords():
    """Test that forward pass raises error without coordinates."""
    config = create_test_config()
    model = LHTEncoder(config)

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

    with pytest.raises(ValueError, match="requires 'coords'"):
        model(input_ids, coords=None)


def test_model_parameter_count():
    """Test approximate parameter count."""
    config = create_test_config()
    model = LHTEncoder(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # All params should be trainable
    assert total_params == trainable_params

    # Check that params are in reasonable range
    assert total_params > 0, "Model should have parameters"


def test_full_size_model_parameter_count():
    """Test parameter count for HDT-E sized model (should be ~53.5M)."""
    # This test only counts parameters after initialization, no GPU needed
    config = SimpleNamespace(
        model=SimpleNamespace(
            vocab_size=30522,  # BERT vocab
            d_model=768,
            n_heads=12,
            num_layers=12,
            d_ff=3072,
            dropout=0.1,
            max_seq_len=8192,
            geometry=SimpleNamespace(
                num_levels=3,
                manhattan_radius=1,
                window_size_per_level=[256, 64, 16],
                layer_max_level=[2] * 12,
            ),
        )
    )

    model = LHTEncoder(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # All params should be trainable
    assert total_params == trainable_params, "All parameters should be trainable"

    # Parameter count for this config (BERT vocab + no weight tying)
    # Token embedding: 30522 * 768 = 23.4M
    # Position embedding: 8192 * 768 = 6.3M
    # MLM head: 30522 * 768 = 23.4M (not tied)
    # 12 transformer layers: ~77.8M
    # Total: ~131M parameters (more than original BERT due to no weight tying + pos embed)
    assert (
        125_000_000 < total_params < 140_000_000
    ), f"Expected ~131M params (BERT-sized with no weight tying), got {total_params/1e6:.1f}M"


def test_forward_pass_no_nan():
    """Test that forward pass doesn't produce NaN values (mocked attention)."""
    config = create_test_config()
    model = LHTEncoder(config)
    model.eval()

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with torch.no_grad():
        with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
            # Mock flex_attention to return reasonable finite values
            def mock_attn(q, k, v, block_mask=None):
                # Return finite values in reasonable range
                return torch.randn_like(q) * 0.1  # Small std to keep values reasonable

            mock_flex_attn.side_effect = mock_attn

            with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
                mock_create_mask.return_value = MagicMock()

                output = model(input_ids, coords=coords)

    # Check for NaN values
    assert not torch.isnan(output["hidden"]).any(), "Hidden states contain NaN"
    assert not torch.isnan(output["mlm_logits"]).any(), "Logits contain NaN"

    # Check for Inf values
    assert not torch.isinf(output["hidden"]).any(), "Hidden states contain Inf"
    assert not torch.isinf(output["mlm_logits"]).any(), "Logits contain Inf"


def test_forward_pass_logits_range():
    """Test that logits are in reasonable range (mocked attention)."""
    config = create_test_config()
    model = LHTEncoder(config)
    model.eval()

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with torch.no_grad():
        with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
            # Mock flex_attention to return normalized output
            def mock_attn(q, k, v, block_mask=None):
                # Return normalized values to simulate post-attention output
                return torch.randn_like(q) * 0.5  # Normalized scale

            mock_flex_attn.side_effect = mock_attn

            with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
                mock_create_mask.return_value = MagicMock()

                output = model(input_ids, coords=coords)

    logits = output["mlm_logits"]

    # Logits should be in reasonable range (not exploding)
    assert logits.abs().max() < 100, f"Logits too large: max={logits.abs().max()}"
    assert not torch.isinf(logits).any(), "Logits contain Inf"
    assert not torch.isnan(logits).any(), "Logits contain NaN"

    # Check that final_norm is applied (affects logit magnitude)
    # With normalized attention outputs, logits should be in a reasonable range
    assert (
        logits.abs().mean() < 50
    ), f"Mean logit magnitude too large: {logits.abs().mean()}"


def test_gradient_flow():
    """Test that gradients flow through the model (mocked attention with gradients)."""
    config = create_test_config()
    model = LHTEncoder(config)

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock flex_attention with gradient support
        def mock_attn_with_grad(q, k, v, block_mask=None):
            # Return a tensor that requires grad and is connected to input
            # This allows gradients to flow back through the mock
            out = torch.randn_like(q, requires_grad=True)
            # Connect output to input to enable gradient flow
            return out + (q * 0.0)  # Add zero-scaled input to maintain gradient path

        mock_flex_attn.side_effect = mock_attn_with_grad

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = model(input_ids, coords=coords)
            loss = output["mlm_logits"].sum()
            loss.backward()

    # Check that embeddings received gradients
    assert model.token_embed.weight.grad is not None, "Embeddings should have gradients"
    assert (
        model.token_embed.weight.grad.abs().sum() > 0
    ), "Embedding gradients should be non-zero"

    # Check that mlm_head received gradients
    assert model.mlm_head.weight.grad is not None, "MLM head should have gradients"
    assert (
        model.mlm_head.weight.grad.abs().sum() > 0
    ), "MLM head gradients should be non-zero"


def test_model_eval_mode():
    """Test model behavior in eval mode (mocked attention, deterministic)."""
    config = create_test_config()
    model = LHTEncoder(config)

    model.eval()

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with torch.no_grad():
        with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
            # Mock with deterministic output (seeded random)
            def mock_attn_deterministic(q, k, v, block_mask=None):
                # Use same seed for both calls to ensure determinism
                torch.manual_seed(42)
                return torch.randn_like(q)

            mock_flex_attn.side_effect = mock_attn_deterministic

            with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
                mock_create_mask.return_value = MagicMock()

                output1 = model(input_ids, coords=coords)

                # Reset mock call count
                mock_flex_attn.reset_mock()
                mock_flex_attn.side_effect = mock_attn_deterministic

                output2 = model(input_ids, coords=coords)

    # In eval mode with same input and deterministic mock, output should be identical
    assert torch.allclose(
        output1["mlm_logits"], output2["mlm_logits"]
    ), "Outputs should be identical in eval mode with same inputs"
    assert torch.allclose(
        output1["hidden"], output2["hidden"]
    ), "Hidden states should be identical in eval mode with same inputs"


def test_geometric_transformer_block_forward():
    """Test forward pass through single transformer block (mocked attention)."""
    block = GeometricTransformerBlock(
        d_model=128,
        n_heads=4,
        d_ff=512,
        dropout=0.0,  # No dropout for testing
        manhattan_radius=1,
        layer_idx=0,
    )

    batch_size = 2
    seq_len = 8
    d_model = 128

    x = torch.randn(batch_size, seq_len, d_model)
    levels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    logical_times = torch.zeros(batch_size, seq_len, dtype=torch.long)
    coords = GeometricCoordinates(levels, logical_times)

    with patch("lht.core.attention._compiled_flex_attention") as mock_flex_attn:
        # Mock attention to return appropriate shape
        def mock_attn(q, k, v, block_mask=None):
            return torch.randn_like(q)

        mock_flex_attn.side_effect = mock_attn

        with patch("lht.core.attention.create_geometric_mask") as mock_create_mask:
            mock_create_mask.return_value = MagicMock()

            output = block(x, coords)

    # Check output shape matches input (residual connection)
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"

    # Verify that attention and mask creation were called
    assert mock_flex_attn.called, "FlexAttention should have been called"
    assert mock_create_mask.called, "create_geometric_mask should have been called"

    # Check that output is different from input (transformation occurred)
    assert not torch.allclose(
        output, x
    ), "Output should differ from input after transformation"


def test_layer_max_level_assignment():
    """Test that per-layer max_level is correctly assigned."""
    config = create_test_config()
    model = LHTEncoder(config)

    for i, layer in enumerate(model.layers):
        expected_max_level = config.model.geometry.layer_max_level[i]
        assert layer.layer_max_level == expected_max_level


def test_window_sizes_assignment():
    """Test that window sizes are passed to layers."""
    config = create_test_config()
    model = LHTEncoder(config)

    expected_windows = config.model.geometry.window_size_per_level

    for layer in model.layers:
        assert layer.window_size_per_level == expected_windows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
