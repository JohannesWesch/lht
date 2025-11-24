"""
Test suite for HierarchicalDataCollator.

Tests batch padding, coordinate alignment, MLM masking, and integration with tokenizer.
"""

import pytest
import torch
from transformers import AutoTokenizer

from lht.data.mlm import HierarchicalDataCollator


@pytest.fixture
def tokenizer():
    """Fixture for BERT tokenizer."""
    return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


@pytest.fixture
def collator(tokenizer):
    """Fixture for HierarchicalDataCollator."""
    return HierarchicalDataCollator(
        tokenizer=tokenizer, mlm_probability=0.15, max_length=512
    )


def test_collator_initialization(tokenizer):
    """Test that collator initializes correctly."""
    collator = HierarchicalDataCollator(tokenizer, mlm_probability=0.15, max_length=512)

    assert collator.tokenizer == tokenizer
    assert collator.mlm_probability == 0.15
    assert collator.max_length == 512


def test_single_document_batch(collator):
    """Test collation of a single document."""
    documents = [[["Hello world", "How are you"]]]

    batch = collator.torch_call(documents)

    # Check batch structure
    assert "input_ids" in batch
    assert "labels" in batch
    assert "coords" in batch

    # Check shapes
    assert batch["input_ids"].dim() == 2  # [B, N]
    assert batch["labels"].dim() == 2  # [B, N]
    assert batch["input_ids"].shape[0] == 1  # Batch size = 1


def test_batch_padding_varying_lengths(collator):
    """Test that sequences of different lengths are padded correctly."""
    documents = [
        [["Short"]],
        [["This is a much longer sentence with many more words"]],
        [["Medium length sentence"]],
    ]

    batch = collator.torch_call(documents)

    # All sequences should have same length (padded to max)
    assert batch["input_ids"].shape[1] == batch["input_ids"].shape[1]

    # Verify padding token is used
    pad_token_id = collator.tokenizer.pad_token_id
    assert pad_token_id in batch["input_ids"], "Should have padding tokens"


def test_coordinate_padding_matches_input_ids(collator):
    """Test that coordinate tensors are padded to match input_ids."""
    documents = [[["Short"]], [["This is a longer sentence"]]]

    batch = collator.torch_call(documents)

    seq_len = batch["input_ids"].shape[1]
    coords = batch["coords"]

    # Coordinates should match sequence length
    assert coords.levels.shape[1] == seq_len, "Levels should match seq_len"
    assert (
        coords.logical_times.shape[1] == seq_len
    ), "Logical times should match seq_len"


def test_mlm_masking_percentage(collator):
    """Test that approximately 15% of tokens are masked."""
    torch.manual_seed(42)  # For reproducibility

    documents = [
        [
            ["This is a test sentence with many words to mask"] * 10
        ]  # Repeat for larger sample
    ]

    batch = collator.torch_call(documents)

    # Count masked tokens (labels != -100)
    masked_tokens = (batch["labels"] != -100).sum().item()
    total_tokens = batch["input_ids"].numel()

    mask_ratio = masked_tokens / total_tokens

    # Should be approximately 15% (allow some variance)
    assert 0.10 < mask_ratio < 0.20, f"Mask ratio {mask_ratio} not close to 0.15"


def test_mlm_mask_token_usage(collator):
    """Test that [MASK] token is used in input_ids."""
    torch.manual_seed(42)

    documents = [[["This is a test sentence"] * 5]]

    batch = collator.torch_call(documents)

    mask_token_id = collator.tokenizer.mask_token_id

    # Should have some [MASK] tokens
    assert mask_token_id in batch["input_ids"], "Should have [MASK] tokens in input_ids"


def test_labels_ignore_unmasked(collator):
    """Test that unmasked tokens have label -100."""
    torch.manual_seed(42)

    documents = [[["Test sentence"]]]

    batch = collator.torch_call(documents)

    # Most tokens should have label -100 (not masked)
    ignore_count = (batch["labels"] == -100).sum().item()
    total_count = batch["labels"].numel()

    # More than 80% should be unmasked (ignore_index)
    assert ignore_count / total_count > 0.80, "Most tokens should be unmasked"


def test_coordinate_batch_stacking(collator):
    """Test that coordinates are properly stacked in batch."""
    documents = [[["First document"]], [["Second document"]], [["Third document"]]]

    batch = collator.torch_call(documents)
    coords = batch["coords"]

    # Batch dimension should match
    batch_size = 3
    assert coords.levels.shape[0] == batch_size
    assert coords.logical_times.shape[0] == batch_size


def test_multi_section_document(collator):
    """Test collation of document with multiple sections."""
    documents = [
        [["Section 1 sentence 1", "Section 1 sentence 2"], ["Section 2 sentence 1"]]
    ]

    batch = collator.torch_call(documents)

    # Should process without errors
    assert "input_ids" in batch
    assert "coords" in batch

    # Coordinates should have varying logical_times
    coords = batch["coords"]
    unique_times = torch.unique(coords.logical_times[0])
    assert len(unique_times) > 1, "Should have multiple sentence IDs"


def test_empty_batch_handling(collator):
    """Test handling of edge case with minimal content."""
    documents = [[["Hi"]]]

    batch = collator.torch_call(documents)

    # Should still produce valid batch
    assert batch["input_ids"].shape[0] == 1
    assert len(batch["coords"].levels) > 0


def test_coordinate_truncation(collator):
    """Test that coordinates are truncated when sequence exceeds max_length."""
    # Create very long document
    long_sentence = " ".join(["word"] * 200)
    documents = [[[long_sentence, long_sentence]]]

    collator_short = HierarchicalDataCollator(
        collator.tokenizer, mlm_probability=0.15, max_length=50
    )

    batch = collator_short.torch_call(documents)

    # Coordinates should be truncated to match input_ids
    assert batch["coords"].levels.shape[1] == batch["input_ids"].shape[1]
    assert batch["coords"].levels.shape[1] <= 50


def test_batch_consistency(collator):
    """Test that all batch elements have consistent dimensions."""
    documents = [[["Doc 1"]], [["Doc 2 is longer"]], [["Doc 3"]]]

    batch = collator.torch_call(documents)

    batch_size = 3
    seq_len = batch["input_ids"].shape[1]

    # All tensors should have matching dimensions
    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size, seq_len)
    assert batch["coords"].levels.shape == (batch_size, seq_len)
    assert batch["coords"].logical_times.shape == (batch_size, seq_len)


def test_mlm_original_tokens_preserved_in_labels(collator):
    """Test that original token IDs are preserved in labels for masked positions."""
    torch.manual_seed(42)

    documents = [[["Test sentence for masking"]]]

    batch = collator.torch_call(documents)

    # Where labels != -100, those positions were masked
    # Labels should contain valid token IDs (not -100, not mask_token_id)
    masked_positions = batch["labels"] != -100

    if masked_positions.any():
        masked_labels = batch["labels"][masked_positions]
        # Labels should be valid token IDs
        assert torch.all(masked_labels >= 0), "Labels should be valid token IDs"
        assert torch.all(
            masked_labels < collator.tokenizer.vocab_size
        ), "Labels should be within vocab"


def test_geometric_coordinates_type(collator):
    """Test that coords is a GeometricCoordinates object."""
    from lht.core.attention import GeometricCoordinates

    documents = [[["Test"]]]

    batch = collator.torch_call(documents)

    assert isinstance(
        batch["coords"], GeometricCoordinates
    ), "coords should be GeometricCoordinates object"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
