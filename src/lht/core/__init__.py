"""
Core geometric attention functionality - essentials only.

This module contains the minimal implementation (~150 lines total):
- GeometricCoordinates: coordinate representation
- geometric_attention(): the attention function
- build_coords(): universal coordinate builder
"""

from .attention import GeometricCoordinates, create_geometric_mask, geometric_attention
from .coords import build_coords
from .model import GeometricTransformerBlock

__all__ = [
    "GeometricCoordinates",
    "create_geometric_mask",
    "geometric_attention",
    "build_coords",
    "GeometricTransformerBlock",
]
