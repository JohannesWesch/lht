"""
Learned Hierarchical Transformer (LHT).

Clean separation: core/ and utils/

CORE (~150 lines):
    - core.attention: Geometric attention mechanism
    - core.coords: Universal coordinate builder
    - core.model: Transformer block (optional)

UTILS (extras):
    - utils.coord_builders: Convenience builders
    - utils.verification: Debug helpers
    - utils.visualization: Simple visualization
    - utils.full_encoder: Complete encoder (optional)
"""

# ============================================================================
# CORE (Essential functionality)
# ============================================================================
from .core import (
    GeometricCoordinates,
    GeometricTransformerBlock,
    build_coords,
    create_geometric_mask,
    geometric_attention,
)

# ============================================================================
# LEGACY (kept for backwards compatibility)
# ============================================================================
from .routers import LevelRouter

# ============================================================================
# UTILS (Convenience and debugging)
# ============================================================================
from .utils import (
    GeometricLHTEncoder,
    build_flat_coords,
    build_three_level_coords,
    build_two_level_coords,
    verify_parent_child_distances,
    visualize_coords,
)

__all__ = [
    # Core (essentials)
    "GeometricCoordinates",
    "geometric_attention",
    "create_geometric_mask",
    "build_coords",
    "GeometricTransformerBlock",
    # Utils (extras)
    "build_two_level_coords",
    "build_three_level_coords",
    "build_flat_coords",
    "verify_parent_child_distances",
    "visualize_coords",
    "GeometricLHTEncoder",
    # Legacy
    "LevelRouter",
]
