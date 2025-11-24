"""
Utilities for geometric attention - convenience and debugging helpers.

This module contains optional helpers:
- Convenience coordinate builders (two_level, three_level)
- Verification functions
- Visualization functions
- Full encoder with embeddings
"""

from .coord_builders import (
    build_flat_coords,
    build_three_level_coords,
    build_two_level_coords,
)
from .full_encoder import GeometricLHTEncoder
from .verification import verify_parent_child_distances
from .visualization import visualize_coords

__all__ = [
    "build_two_level_coords",
    "build_three_level_coords",
    "build_flat_coords",
    "verify_parent_child_distances",
    "visualize_coords",
    "GeometricLHTEncoder",
]
