"""
Test suite for geometric coordinate builders.

Verifies the core guarantee: every parent-child pair has Manhattan distance = 1
across hierarchies of any depth.
"""

import pytest

from lht import (
    build_hierarchy_coordinates_simple,
    build_three_level_coordinates,
    build_two_level_coordinates,
    verify_parent_child_distances,
)


def test_two_level_parent_child_distance():
    """Test that tokens and sentences have distance = 1."""
    num_tokens = 6
    sentence_boundaries = [2, 5]  # 2 sentences

    coords = build_two_level_coordinates(num_tokens, sentence_boundaries)

    # Define parent-child pairs
    parent_child_pairs = [
        (6, 0),
        (6, 1),
        (6, 2),  # SENT_0 → tokens 0-2
        (7, 3),
        (7, 4),
        (7, 5),  # SENT_1 → tokens 3-5
    ]

    stats = verify_parent_child_distances(
        coords, parent_child_pairs, max_expected_distance=1
    )

    assert (
        stats["num_violations"] == 0
    ), "All parent-child pairs should have distance ≤ 1"
    assert stats["max_distance"] == 1, "Max distance should be exactly 1"
    assert abs(stats["avg_distance"] - 1.0) < 0.001, "Avg distance should be 1.0"


def test_three_level_parent_child_distance():
    """Test that tokens → sentences → sections all have distance = 1."""
    num_tokens = 9
    sentence_boundaries = [2, 5, 8]
    section_boundaries = [1]  # section 0: sents 0-1, section 1: sent 2

    coords = build_three_level_coordinates(
        num_tokens, sentence_boundaries, section_boundaries
    )

    parent_child_pairs = [
        # Tokens → Sentences
        (9, 0),
        (9, 1),
        (9, 2),  # SENT_0 → tokens
        (10, 3),
        (10, 4),
        (10, 5),  # SENT_1 → tokens
        (11, 6),
        (11, 7),
        (11, 8),  # SENT_2 → tokens
        # Sentences → Sections
        (12, 9),
        (12, 10),  # SEC_0 → SENT_0, SENT_1
        (13, 11),  # SEC_1 → SENT_2
    ]

    stats = verify_parent_child_distances(
        coords, parent_child_pairs, max_expected_distance=1
    )

    assert (
        stats["num_violations"] == 0
    ), "All parent-child pairs should have distance ≤ 1"
    assert stats["max_distance"] == 1, "Max distance should be exactly 1"


def test_five_level_universal():
    """Test universal builder with 5 levels."""
    num_nodes_per_level = [10, 4, 3, 2, 1]  # tokens → sents → paras → secs → chaps

    parent_maps = [
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],  # tokens
        [0, 0, 1, 2],  # sentences
        [0, 0, 1],  # paragraphs
        [0, 0],  # sections
        [0],  # chapters
    ]

    coords = build_hierarchy_coordinates_simple(num_nodes_per_level, parent_maps)

    # Test sample parent-child pairs across all levels
    parent_child_pairs = [
        # Level 0 → 1
        (10, 0),
        (10, 1),
        # Level 1 → 2
        (14, 10),
        # Level 2 → 3
        (17, 14),
        # Level 3 → 4
        (19, 17),
    ]

    stats = verify_parent_child_distances(
        coords, parent_child_pairs, max_expected_distance=1
    )

    assert (
        stats["num_violations"] == 0
    ), "All parent-child pairs should have distance ≤ 1"
    assert stats["max_distance"] == 1, "Max distance should be exactly 1"


def test_seven_level_universal():
    """Test that universal builder scales to 7 levels."""
    num_nodes_per_level = [8, 4, 2, 2, 1, 1, 1]

    parent_maps = [
        [0, 0, 1, 1, 2, 2, 3, 3],  # level 0
        [0, 0, 1, 1],  # level 1
        [0, 1],  # level 2
        [0, 0],  # level 3
        [0],  # level 4
        [0],  # level 5
        [0],  # level 6 (top)
    ]

    coords = build_hierarchy_coordinates_simple(num_nodes_per_level, parent_maps)

    # Test one pair from each level transition
    parent_child_pairs = [
        (8, 0),  # level 1 → 0
        (12, 8),  # level 2 → 1
        (14, 12),  # level 3 → 2
        (16, 14),  # level 4 → 3
        (17, 16),  # level 5 → 4
        (18, 17),  # level 6 → 5
    ]

    stats = verify_parent_child_distances(
        coords, parent_child_pairs, max_expected_distance=1
    )

    assert (
        stats["num_violations"] == 0
    ), "All parent-child pairs should have distance ≤ 1"
    assert stats["max_distance"] == 1, "Max distance should be exactly 1"


def test_coordinate_dimensions():
    """Test that coordinate tensors have correct shapes."""
    num_tokens = 6
    sentence_boundaries = [2, 5]

    coords = build_two_level_coordinates(num_tokens, sentence_boundaries)

    expected_len = num_tokens + len(sentence_boundaries)  # tokens + summaries

    assert len(coords.levels) == expected_len
    assert len(coords.logical_times) == expected_len
    assert len(coords.physical_positions) == expected_len


def test_coordinate_values():
    """Test that coordinate values are correctly assigned."""
    num_tokens = 4
    sentence_boundaries = [1, 3]  # sent 0: tokens 0-1, sent 1: tokens 2-3

    coords = build_two_level_coordinates(num_tokens, sentence_boundaries)

    # Tokens should have level=0
    assert all(coords.levels[:4] == 0)

    # Sentences should have level=1
    assert all(coords.levels[4:] == 1)

    # Tokens in sentence 0 should have time=0
    assert coords.logical_times[0] == 0
    assert coords.logical_times[1] == 0

    # Tokens in sentence 1 should have time=1
    assert coords.logical_times[2] == 1
    assert coords.logical_times[3] == 1

    # Sentence summaries should match their sentence ID
    assert coords.logical_times[4] == 0  # SENT_0
    assert coords.logical_times[5] == 1  # SENT_1


def test_manhattan_distance_computation():
    """Test the core Manhattan distance formula."""
    # Simple 2-token + 1-sentence example
    num_tokens = 2
    sentence_boundaries = [1]

    coords = build_two_level_coordinates(num_tokens, sentence_boundaries)

    # Token 0: level=0, time=0
    # Token 1: level=0, time=0
    # Sent 0: level=1, time=0

    # Distance: token 0 → sentence 0
    dist = abs(coords.levels[0] - coords.levels[2]) + abs(
        coords.logical_times[0] - coords.logical_times[2]
    )
    assert dist == 1, "Token → sentence should have distance = 1"

    # Distance: token 0 → token 1 (same sentence)
    dist = abs(coords.levels[0] - coords.levels[1]) + abs(
        coords.logical_times[0] - coords.logical_times[1]
    )
    assert dist == 0, "Tokens in same sentence should have distance = 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
