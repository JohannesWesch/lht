"""
Learned Hierarchical Transformer (LHT).

This package implements:
- Encoder architecture with local → mid → global layers.
- H-Net-style routers to learn sentence/section boundaries.
- HDT-style hierarchical attention masks using xFormers.
"""

from .model import LHTEncoder

__all__ = ["LHTEncoder"]
