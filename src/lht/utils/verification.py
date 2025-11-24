"""
Verification and debugging utilities.
"""

from ..core import GeometricCoordinates


def verify_parent_child_distances(
    coords: GeometricCoordinates,
    parent_child_pairs: list[tuple[int, int]],
    max_expected_distance: int = 1,
) -> dict:
    """
    Verify that parent-child pairs are within expected Manhattan distance.

    Useful for debugging coordinate assignments.

    Args:
        coords: GeometricCoordinates to check
        parent_child_pairs: list of (parent_idx, child_idx) tuples
        max_expected_distance: maximum expected distance

    Returns:
        dict with statistics and any violations

    Example:
        stats = verify_parent_child_distances(
            coords,
            [(6, 0), (6, 1), (7, 3)],  # parent → child pairs
            max_expected_distance=1,
        )
        print(f"Violations: {stats['num_violations']}")
    """
    violations = []
    distances = []

    for parent_idx, child_idx in parent_child_pairs:
        parent_level = coords.levels[parent_idx].item()
        parent_time = coords.logical_times[parent_idx].item()
        child_level = coords.levels[child_idx].item()
        child_time = coords.logical_times[child_idx].item()

        dist = abs(parent_level - child_level) + abs(parent_time - child_time)
        distances.append(dist)

        if dist > max_expected_distance:
            violations.append(
                {
                    "parent_idx": parent_idx,
                    "child_idx": child_idx,
                    "distance": dist,
                    "parent_coords": (parent_level, parent_time),
                    "child_coords": (child_level, child_time),
                }
            )

    return {
        "num_pairs": len(parent_child_pairs),
        "max_distance": max(distances) if distances else 0,
        "avg_distance": sum(distances) / len(distances) if distances else 0,
        "num_violations": len(violations),
        "violations": violations,
    }
