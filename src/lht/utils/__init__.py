"""
Utilities for Multi-Level Sliding Window Attention (ML-SWA).

This module contains optional helpers:
- Nested document builder (build_coords_from_nested_list)
- Seeding utilities
"""

from .nested_builder import build_coords_from_nested_list
from .seeding import set_seed

__all__ = [
    "build_coords_from_nested_list",
    "set_seed",
]
