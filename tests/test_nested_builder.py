"""
Test suite for nested document to coordinates pipeline.

Tests the build_coords_from_nested_list() function which converts
nested document structure into flat token sequences with geometric coordinates.
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

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    # Should have tokens
    assert len(input_ids) > 0, "Should have some tokens"

    # Coordinates should match input_ids length (one coord per token, no hierarchy nodes)
    assert len(coords.levels) == len(input_ids), "Coords should match input_ids length"

    # All tokens should be at level 0
    assert torch.all(coords.levels == 0), "All tokens should be at level 0"

    # All tokens in same sentence should have same logical_time (sentence_id=0)
    assert torch.all(
        coords.logical_times == 0
    ), "All tokens should have x=0 (sentence 0)"


def test_single_section_multiple_sentences(tokenizer):
    """Test one section with multiple sentences."""
    document = [["Hello world", "How are you", "Fine thanks"]]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    assert len(input_ids) > 0

    # Should have tokens from 3 sentences
    # Each token's logical_time should be its sentence_id
    unique_times = torch.unique(coords.logical_times)
    assert len(unique_times) == 3, "Should have 3 different sentence IDs (0, 1, 2)"


def test_multiple_sections_multiple_sentences(tokenizer):
    """Test multi-section document."""
    document = [
        ["Sentence 1 in section 1", "Sentence 2 in section 1"],
        ["Sentence 1 in section 2", "Sentence 2 in section 2"],
    ]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    assert len(input_ids) > 0

    # Should have 4 sentences total
    unique_times = torch.unique(coords.logical_times)
    assert len(unique_times) == 4, "Should have 4 sentence IDs"


def test_coordinate_assignment_correctness(tokenizer):
    """Test that coordinate assignment follows the hierarchy rules."""
    document = [["First sentence", "Second sentence"], ["Third sentence"]]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    # Manually compute expected structure
    # Section 0: sentences 0, 1
    # Section 1: sentence 2

    # Find token boundaries by checking where logical_time changes
    times = coords.logical_times.tolist()

    # All tokens of first sentence should have time=0
    first_sent_end = times.index(1) if 1 in times else len(times)
    assert all(
        t == 0 for t in times[:first_sent_end]
    ), "First sentence tokens should have x=0"

    # Tokens of second sentence should have time=1
    if 1 in times:
        second_sent_start = first_sent_end
        second_sent_end = times.index(2) if 2 in times else len(times)
        assert all(
            t == 1 for t in times[second_sent_start:second_sent_end]
        ), "Second sentence tokens should have x=1"


def test_max_length_truncation(tokenizer):
    """Test that documents are truncated at max_length."""
    # Create a document that will exceed max_length
    long_sentence = " ".join(["word"] * 100)  # Very long sentence
    document = [[long_sentence, long_sentence, long_sentence]]

    max_length = 50
    input_ids, coords = build_coords_from_nested_list(
        document, tokenizer, max_length=max_length
    )

    # Should be truncated to max_length
    assert len(input_ids) <= max_length, f"Should be truncated to {max_length}"
    assert len(coords.levels) == len(input_ids), "Coords should match truncated length"


def test_empty_sentence_handling(tokenizer):
    """Test handling of empty strings in document."""
    document = [["Hello", "", "World"]]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    # Empty strings should be skipped but not crash
    assert len(input_ids) > 0, "Should have tokens from non-empty sentences"


def test_parent_child_distance_tokens_to_sentences(tokenizer):
    """Verify tokens are assigned to correct sentences via logical_time."""
    document = [["Hello world", "How are you"]]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    # All elements are tokens at level 0 (no separate sentence/section tokens)
    assert torch.all(coords.levels == 0), "All coordinates should be for tokens"

    # Tokens in first sentence have logical_time=0, in second have logical_time=1
    times = coords.logical_times.tolist()

    # Verify at least two different sentence IDs (logical_times)
    assert max(times) >= 1, "Should have at least 2 different sentence IDs"


def test_device_placement(tokenizer):
    """Test that coordinates are placed on correct device."""
    document = [["Hello world"]]

    # Test CPU
    input_ids, coords = build_coords_from_nested_list(
        document, tokenizer, device=torch.device("cpu")
    )
    assert coords.levels.device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        input_ids, coords = build_coords_from_nested_list(
            document, tokenizer, device=torch.device("cuda")
        )
        assert coords.levels.device.type == "cuda"


def test_tokenization_without_special_tokens(tokenizer):
    """Verify that tokenization doesn't add [CLS] or [SEP] tokens."""
    document = [["Hello"]]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

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

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    # Should have 6 sentences total
    unique_times = torch.unique(coords.logical_times)
    assert len(unique_times) == 6, "Should have 6 sentence IDs"


def test_coordinate_consistency(tokenizer):
    """Test that coordinate tensors have consistent lengths."""
    document = [["Hello world", "Test sentence"], ["Another section"]]

    input_ids, coords = build_coords_from_nested_list(document, tokenizer)

    # All coordinate tensors should have same length
    assert len(coords.levels) == len(coords.logical_times)
    assert len(coords.levels) == len(coords.physical_positions)
    # Coords should match input_ids (one coordinate per token)
    assert len(coords.levels) == len(input_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
