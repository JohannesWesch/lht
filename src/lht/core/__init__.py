"""
Core Multi-Level Sliding Window Attention (ML-SWA) - essentials only.

This module contains the minimal implementation:
- HierarchicalPositions: sparse enumeration vectors per level
- mlswa_attention(): cascading per-level attention function
- create_hierarchical_mask(): vectorized mask creation
"""

from .attention import HierarchicalPositions, create_hierarchical_mask, mlswa_attention
from .model import MLSWATransformerBlock

__all__ = [
    "HierarchicalPositions",
    "create_hierarchical_mask",
    "mlswa_attention",
    "MLSWATransformerBlock",
]
