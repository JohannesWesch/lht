"""
Learned Hierarchical Transformer (LHT).

Clean separation: core/ and utils/

CORE:
    - core.attention: Multi-Level Sliding Window Attention (ML-SWA)
    - core.model: MLSWATransformerBlock (single transformer layer)

UTILS (extras):
    - utils.nested_builder: Document structure to hierarchical positions
    - utils.seeding: Random seed utilities

Main encoder model is in model.py (LHTEncoder).
"""

# ============================================================================
# CONFIG (Configuration management)
# ============================================================================
from .config import (
    DataConfig,
    ExperimentConfig,
    MLSWAConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
)

# ============================================================================
# CORE (Essential functionality)
# ============================================================================
from .core import (
    HierarchicalPositions,
    MLSWATransformerBlock,
    create_hierarchical_mask,
    mlswa_attention,
)

# ============================================================================
# UTILS (Convenience and debugging)
# ============================================================================
from .utils import build_coords_from_nested_list, set_seed

__all__ = [
    # Core (essentials)
    "HierarchicalPositions",
    "mlswa_attention",
    "create_hierarchical_mask",
    "MLSWATransformerBlock",
    # Config
    "MLSWAConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "WandbConfig",
    "ExperimentConfig",
    "load_config",
    # Utils (extras)
    "build_coords_from_nested_list",
    "set_seed",
]
