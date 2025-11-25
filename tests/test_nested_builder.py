"""
Test suite for nested document to hierarchical positions pipeline.

Tests the build_coords_from_nested_list() function which converts
nested document structure into flat token sequences with sparse enumeration vectors.
"""

import pytest
import torch
from transformers import AutoTokenizer

from lht.utils.nested_builder import build_coords_from_nested_list


@pytest.fixture
def tokenizer():
    """Fixture for BERT tokenizer."""
    return AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


def test_single_section_single_sentence(tokenizer):
    """Test simplest case: one section with one sentence."""
    document = [["Hello world"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # Should have tokens
    assert len(input_ids) > 0, "Should have some tokens"

    # Positions should have 3 levels
    assert positions.num_levels == 3, "Should have 3 hierarchy levels"

    # Level 0: all tokens should be enumerated (1, 2, 3, ...)
    level_0 = positions.level_enums[0]
    assert len(level_0) == len(input_ids), "Level 0 should match input_ids length"
    assert torch.all(
        level_0 > 0
    ), "All tokens should have non-zero enumeration at level 0"

    # Level 1: only last token should be enumerated (sentence boundary)
    level_1 = positions.level_enums[1]
    assert torch.sum(level_1 > 0) == 1, "Should have 1 sentence boundary"
    assert level_1[-1] > 0, "Last token should be sentence boundary"

    # Level 2: last token should also be section boundary
    level_2 = positions.level_enums[2]
    assert torch.sum(level_2 > 0) == 1, "Should have 1 section boundary"
    assert level_2[-1] > 0, "Last token should be section boundary"


def test_single_section_multiple_sentences(tokenizer):
    """Test one section with multiple sentences."""
    document = [["Hello world", "How are you", "Fine thanks"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    assert len(input_ids) > 0

    # Should have 3 sentence boundaries (level 1)
    level_1 = positions.level_enums[1]
    num_sentence_boundaries = torch.sum(level_1 > 0).item()
    assert num_sentence_boundaries == 3, "Should have 3 sentence boundaries"

    # Should have sequential enumeration at boundaries: 1, 2, 3
    boundary_values = level_1[level_1 > 0]
    assert torch.all(
        boundary_values == torch.tensor([1, 2, 3])
    ), "Should have sequential sentence IDs"


def test_multiple_sections_multiple_sentences(tokenizer):
    """Test multi-section document."""
    document = [
        ["Sentence 1 in section 1", "Sentence 2 in section 1"],
        ["Sentence 1 in section 2", "Sentence 2 in section 2"],
    ]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    assert len(input_ids) > 0

    # Should have 4 sentence boundaries (level 1)
    level_1 = positions.level_enums[1]
    num_sentence_boundaries = torch.sum(level_1 > 0).item()
    assert num_sentence_boundaries == 4, "Should have 4 sentence boundaries"

    # Should have 2 section boundaries (level 2)
    level_2 = positions.level_enums[2]
    num_section_boundaries = torch.sum(level_2 > 0).item()
    assert num_section_boundaries == 2, "Should have 2 section boundaries"


def test_coordinate_assignment_correctness(tokenizer):
    """Test that enumeration assignment follows the hierarchy rules."""
    document = [["First sentence", "Second sentence"], ["Third sentence"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # Expected structure:
    # Section 0: sentences 1, 2 (at end of each sentence)
    # Section 1: sentence 3 (at end)

    level_1 = positions.level_enums[1]
    level_2 = positions.level_enums[2]

    # Should have 3 sentence boundaries
    num_sentence_boundaries = torch.sum(level_1 > 0).item()
    assert num_sentence_boundaries == 3, "Should have 3 sentence boundaries"

    # Should have 2 section boundaries
    num_section_boundaries = torch.sum(level_2 > 0).item()
    assert num_section_boundaries == 2, "Should have 2 section boundaries"

    # Section boundaries should be at sentence boundaries
    section_boundary_indices = (level_2 > 0).nonzero(as_tuple=True)[0]
    for idx in section_boundary_indices:
        assert level_1[idx] > 0, "Section boundaries should also be sentence boundaries"


def test_max_length_truncation(tokenizer):
    """Test that documents are truncated at max_length."""
    # Create a document that will exceed max_length
    long_sentence = " ".join(["word"] * 100)  # Very long sentence
    document = [[long_sentence, long_sentence, long_sentence]]

    max_length = 50
    input_ids, positions = build_coords_from_nested_list(
        document, tokenizer, max_length=max_length
    )

    # Should be truncated to max_length
    assert len(input_ids) <= max_length, f"Should be truncated to {max_length}"
    assert len(positions.level_enums[0]) == len(
        input_ids
    ), "Positions should match truncated length"


def test_empty_sentence_handling(tokenizer):
    """Test handling of empty strings in document."""
    document = [["Hello", "", "World"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # Empty strings should be skipped but not crash
    assert len(input_ids) > 0, "Should have tokens from non-empty sentences"


def test_hierarchical_enumeration(tokenizer):
    """Verify hierarchical enumeration is correct."""
    document = [["Hello world", "How are you"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # Level 0: all tokens should have sequential enumeration
    level_0 = positions.level_enums[0]
    assert len(level_0) == len(
        input_ids
    ), "All positions should be enumerated at level 0"
    assert torch.all(level_0 > 0), "All level 0 enumerations should be positive"

    # Level 1: should have 2 sentence boundaries
    level_1 = positions.level_enums[1]
    num_sentence_boundaries = torch.sum(level_1 > 0).item()
    assert num_sentence_boundaries == 2, "Should have 2 sentence boundaries"


def test_device_placement(tokenizer):
    """Test that positions are placed on correct device."""
    document = [["Hello world"]]

    # Test CPU
    input_ids, positions = build_coords_from_nested_list(
        document, tokenizer, device=torch.device("cpu")
    )
    assert positions.level_enums[0].device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        input_ids, positions = build_coords_from_nested_list(
            document, tokenizer, device=torch.device("cuda")
        )
        assert positions.level_enums[0].device.type == "cuda"


def test_tokenization_without_special_tokens(tokenizer):
    """Verify that tokenization doesn't add [CLS] or [SEP] tokens."""
    document = [["Hello"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # Should not have [CLS] (101) or [SEP] (102) tokens for BERT
    assert 101 not in input_ids, "Should not have [CLS] token"
    assert 102 not in input_ids, "Should not have [SEP] token"


def test_varying_section_sizes(tokenizer):
    """Test document with sections of different sizes."""
    document = [
        ["One sentence"],
        ["First", "Second", "Third", "Fourth"],
        ["Single"],
    ]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # Should have 6 sentence boundaries total (level 1)
    level_1 = positions.level_enums[1]
    num_sentence_boundaries = torch.sum(level_1 > 0).item()
    assert num_sentence_boundaries == 6, "Should have 6 sentence boundaries"


def test_position_consistency(tokenizer):
    """Test that position tensors have consistent lengths."""
    document = [["Hello world", "Test sentence"], ["Another section"]]

    input_ids, positions = build_coords_from_nested_list(document, tokenizer)

    # All level enumeration tensors should have same length
    for i in range(positions.num_levels):
        assert len(positions.level_enums[i]) == len(
            input_ids
        ), f"Level {i} should match input_ids length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
