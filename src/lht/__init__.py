"""
Learned Hierarchical Transformer (LHT).

This package implements:
- Encoder architecture with local → mid → global layers
- H-Net-style routers to learn K-level hierarchies (variable depth)
- HDT-style hierarchical attention masks using xFormers
- Support for 2, 3, 4, or N abstraction levels

Example hierarchies:
- 2 levels: tokens → sentences
- 3 levels: tokens → sentences → sections
- 4 levels: tokens → sentences → paragraphs → sections
"""

from .hierarchy import HierarchyManager
from .model import LHTEncoder
from .routers import LevelRouter

__all__ = ["LHTEncoder", "HierarchyManager", "LevelRouter"]
