"""
Simple visualization utilities for debugging.
"""

import io

from ..core import GeometricCoordinates


def visualize_coords(
    coords: GeometricCoordinates,
    radius: int,
    token_names: list = None,
) -> str:
    """
    Simple text visualization of coordinates and attention pattern.

    Args:
        coords: GeometricCoordinates to visualize
        radius: Manhattan distance radius
        token_names: optional list of token names

    Returns:
        String visualization

    Example:
        print(visualize_coords(coords, radius=2, token_names=["Hello", "world", ...]))
    """
    output = io.StringIO()

    output.write("=" * 60 + "\n")
    output.write("Geometric Coordinates\n")
    output.write(f"Radius: {radius}\n")
    output.write("=" * 60 + "\n\n")

    output.write(f"{'Token':<20} | Level | Logical Time\n")
    output.write("-" * 50 + "\n")

    for i in range(len(coords.levels)):
        token_name = token_names[i] if token_names else f"token_{i}"
        level = coords.levels[i].item()
        time = coords.logical_times[i].item()

        output.write(f"{token_name:<20} | {level:5d} | {time:12d}\n")

    output.write("\n" + "=" * 60 + "\n")

    return output.getvalue()
