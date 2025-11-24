"""
Test that sentence summaries are correctly included in SWA via geometric coordinates.
"""

import sys

sys.path.insert(0, "/home/ka/ka_stud/ka_upygb/repos/lht/src")

from lht.core.coords import build_coords


def test_parent_child_distance_one():
    """Verify all parent-child pairs have Manhattan distance = 1."""
    parent_maps = [
        [0, 0, 0, 1, 1],  # 5 tokens: 3 in sent 0, 2 in sent 1
        [0, 1],  # 2 sentences: IDs (top level)
    ]
    coords = build_coords(parent_maps)

    # Token 0 belongs to sentence 0
    # Token 0: coords[0], Sentence 0: coords[5]
    token_0_level = coords.levels[0].item()
    token_0_x = coords.logical_times[0].item()
    sent_0_level = coords.levels[5].item()
    sent_0_x = coords.logical_times[5].item()

    dist = abs(token_0_level - sent_0_level) + abs(token_0_x - sent_0_x)
    assert dist == 1, f"Token 0 → Sentence 0 distance should be 1, got {dist}"

    # Token 3 belongs to sentence 1
    token_3_level = coords.levels[3].item()
    token_3_x = coords.logical_times[3].item()
    sent_1_level = coords.levels[6].item()
    sent_1_x = coords.logical_times[6].item()

    dist = abs(token_3_level - sent_1_level) + abs(token_3_x - sent_1_x)
    assert dist == 1, f"Token 3 → Sentence 1 distance should be 1, got {dist}"


def test_cross_sentence_distance():
    """Verify cross-sentence distances."""
    parent_maps = [
        [0, 0, 0, 1, 1],
        [0, 1],
    ]
    coords = build_coords(parent_maps)

    # Token 0 (sentence 0, x=0) → Sentence 1 (x=1)
    token_0_level = coords.levels[0].item()
    token_0_x = coords.logical_times[0].item()
    sent_1_level = coords.levels[6].item()
    sent_1_x = coords.logical_times[6].item()

    dist = abs(token_0_level - sent_1_level) + abs(token_0_x - sent_1_x)
    assert (
        dist == 2
    ), f"Cross-sentence distance should be 2 (|Δx|=1, |Δy|=1), got {dist}"


def test_sliding_window_includes_parent():
    """Test that radius=1 includes parent."""
    parent_maps = [
        [0, 0, 0, 1, 1],
        [0, 1],
    ]
    coords = build_coords(parent_maps)
    radius = 1

    # Token 0 should attend to Sentence 0 (its parent)
    token_idx = 0
    sent_0_idx = 5
    dist = abs(coords.levels[token_idx] - coords.levels[sent_0_idx]) + abs(
        coords.logical_times[token_idx] - coords.logical_times[sent_0_idx]
    )
    assert dist <= radius, "Token 0 should attend to its parent Sentence 0"

    # Token 0 with radius=1 can attend to Sentence 1 (distance=2 requires radius ≥ 2)
    sent_1_idx = 6
    dist = abs(coords.levels[token_idx] - coords.levels[sent_1_idx]) + abs(
        coords.logical_times[token_idx] - coords.logical_times[sent_1_idx]
    )
    assert dist == 2, f"Token 0 → Sentence 1 distance should be 2, got {dist}"


def test_three_level_hierarchy():
    """Test coordinate assignment for 3-level hierarchy."""
    parent_maps = [
        [0, 0, 0, 1, 1],  # 5 tokens in 2 sentences
        [0, 0],  # 2 sentences in 1 paragraph
        [0],  # 1 paragraph
    ]
    coords = build_coords(parent_maps)

    # Verify parent-child distances at all levels
    # Token 0 → Sentence 0
    dist_01 = abs(coords.levels[0] - coords.levels[5]) + abs(
        coords.logical_times[0] - coords.logical_times[5]
    )
    assert dist_01 == 1, "Token → Sentence distance should be 1"

    # Sentence 0 → Paragraph 0
    dist_12 = abs(coords.levels[5] - coords.levels[7]) + abs(
        coords.logical_times[5] - coords.logical_times[7]
    )
    assert dist_12 == 1, "Sentence → Paragraph distance should be 1"


def test_siblings_share_x_coordinate():
    """Verify tokens in same group share x-coordinate (logical_time = parent_id)."""
    parent_maps = [
        [0, 0, 0, 1, 1],
        [0, 1],  # Top level: sentence IDs
    ]
    coords = build_coords(parent_maps)

    # Tokens in sentence 0 should all have x=0
    for i in range(3):
        assert (
            coords.logical_times[i].item() == 0
        ), f"Token {i} (in sent 0) should have x=0, got {coords.logical_times[i].item()}"

    # Tokens in sentence 1 should all have x=1
    for i in range(3, 5):
        assert (
            coords.logical_times[i].item() == 1
        ), f"Token {i} (in sent 1) should have x=1, got {coords.logical_times[i].item()}"


def test_parent_has_same_x_as_children():
    """Verify parent summaries share x with their children."""
    parent_maps = [
        [0, 0, 0, 1, 1],
        [0, 1],  # Sentence IDs
    ]
    coords = build_coords(parent_maps)

    # Sentence 0 (ID=0) should have x=0 (same as its tokens)
    sent_0_x = coords.logical_times[5].item()
    assert sent_0_x == 0, f"Sentence 0 should have x=0, got {sent_0_x}"

    # Sentence 1 (ID=1) should have x=1 (same as its tokens)
    sent_1_x = coords.logical_times[6].item()
    assert sent_1_x == 1, f"Sentence 1 should have x=1, got {sent_1_x}"


if __name__ == "__main__":
    test_parent_child_distance_one()
    test_cross_sentence_distance()
    test_sliding_window_includes_parent()
    test_three_level_hierarchy()
    test_siblings_share_x_coordinate()
    test_parent_has_same_x_as_children()
    print("✅ All coordinate design tests passed!")
