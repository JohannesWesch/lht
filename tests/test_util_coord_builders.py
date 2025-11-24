"""
Test that utility coordinate builders produce correct hierarchical structure.
"""

import sys

import torch

sys.path.insert(0, "/home/ka/ka_stud/ka_upygb/repos/lht/src")

from lht.utils import (
    build_flat_coords,
    build_three_level_coords,
    build_two_level_coords,
)


def test_two_level_coords():
    """Test 2-level coordinate builder."""
    # 6 tokens, sentences end at positions 2 and 5
    # Sentence 0: tokens 0,1,2
    # Sentence 1: tokens 3,4,5
    coords = build_two_level_coords(6, [2, 5])

    # Verify tokens have logical_time = sentence_id
    for i in range(3):
        assert coords.levels[i] == 0, f"Token {i} should be at level 0"
        assert coords.logical_times[i] == 0, f"Token {i} (sent 0) should have x=0"
    for i in range(3, 6):
        assert coords.levels[i] == 0, f"Token {i} should be at level 0"
        assert coords.logical_times[i] == 1, f"Token {i} (sent 1) should have x=1"

    # Verify sentence summaries have logical_time = own_id (top level)
    assert coords.levels[6] == 1, "Sentence 0 should be at level 1"
    assert (
        coords.logical_times[6] == 0
    ), "Sentence 0 should have x=0 (same as its tokens)"

    assert coords.levels[7] == 1, "Sentence 1 should be at level 1"
    assert (
        coords.logical_times[7] == 1
    ), "Sentence 1 should have x=1 (same as its tokens)"

    print("✅ Two-level coords test passed")


def test_three_level_coords():
    """Test 3-level coordinate builder."""
    # 9 tokens, 3 sentences, 2 sections
    # Sentence boundaries: [2, 5, 8] → sentences end at these positions
    # Section boundaries: [1] → section 0 contains sentences 0-1, section 1 contains sentence 2
    coords = build_three_level_coords(9, [2, 5, 8], [1])

    # Verify tokens have logical_time = sentence_id
    for i in range(3):  # Tokens 0-2 in sentence 0
        assert coords.levels[i] == 0, f"Token {i} should be at level 0"
        assert coords.logical_times[i] == 0, f"Token {i} (sent 0) should have x=0"
    for i in range(3, 6):  # Tokens 3-5 in sentence 1
        assert coords.levels[i] == 0, f"Token {i} should be at level 0"
        assert coords.logical_times[i] == 1, f"Token {i} (sent 1) should have x=1"
    for i in range(6, 9):  # Tokens 6-8 in sentence 2
        assert coords.levels[i] == 0, f"Token {i} should be at level 0"
        assert coords.logical_times[i] == 2, f"Token {i} (sent 2) should have x=2"

    # Verify sentences have logical_time = section_id
    assert (
        coords.levels[9] == 1 and coords.logical_times[9] == 0
    ), "Sentence 0 (sec 0) at x=0"
    assert (
        coords.levels[10] == 1 and coords.logical_times[10] == 0
    ), "Sentence 1 (sec 0) at x=0"
    assert (
        coords.levels[11] == 1 and coords.logical_times[11] == 1
    ), "Sentence 2 (sec 1) at x=1"

    # Verify sections have logical_time = own_id (top level)
    assert coords.levels[12] == 2 and coords.logical_times[12] == 0, "Section 0 at x=0"
    assert coords.levels[13] == 2 and coords.logical_times[13] == 1, "Section 1 at x=1"

    print("✅ Three-level coords test passed")


def test_flat_coords():
    """Test flat (no hierarchy) coordinate builder."""
    coords = build_flat_coords(10)

    # All should be at level 0
    assert torch.all(coords.levels == 0), "All tokens should be at level 0"

    # Should have sequential x
    assert torch.all(
        coords.logical_times == torch.arange(10)
    ), "Should have sequential x"

    print("✅ Flat coords test passed")


def test_parent_child_distances():
    """Verify parent-child distances are 1 in all utility builders."""
    # Test 2-level
    coords = build_two_level_coords(6, [2, 5])

    # Token 0 (x=0, y=0) → Sentence 0 (x=0, y=1)
    dist = abs(coords.levels[0] - coords.levels[6]) + abs(
        coords.logical_times[0] - coords.logical_times[6]
    )
    assert dist == 1, f"Token 0 → Sentence 0 should be distance 1, got {dist}"

    # Token 3 (x=1, y=0) → Sentence 1 (x=1, y=1)
    dist = abs(coords.levels[3] - coords.levels[7]) + abs(
        coords.logical_times[3] - coords.logical_times[7]
    )
    assert dist == 1, f"Token 3 → Sentence 1 should be distance 1, got {dist}"

    print("✅ Parent-child distance test passed")


if __name__ == "__main__":
    test_two_level_coords()
    test_three_level_coords()
    test_flat_coords()
    test_parent_child_distances()
    print("\n✅ All utility coord builder tests passed!")
