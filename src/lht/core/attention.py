"""
Core cascading per-level sliding window attention.

Implements sparse hierarchical enumeration with vectorized window checks:
    - Level 0: All tokens (dense enumeration)
    - Level 1+: Boundary positions only (sparse enumeration, 0 = non-participant)
"""

from typing import List

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class HierarchicalPositions:
    """
    Sparse hierarchical enumeration vectors for cascading sliding windows.

    Each level has an enumeration vector [N] or [B, N]:
    - Non-zero values: sequential enumeration for that level
    - Zero values: position doesn't participate in this level

    Example (9 tokens, 4 sentences, 2 sections):
        level_enums[0] = [1,2,3,4,5,6,7,8,9]  # all tokens
        level_enums[1] = [1,0,0,2,0,3,0,0,4]  # sentence boundaries
        level_enums[2] = [1,0,0,0,0,2,0,0,0]  # section boundaries
    """

    def __init__(self, level_enums: List[torch.Tensor]):
        """
        Args:
            level_enums: List of enumeration tensors [N] or [B, N], one per level
        """
        self.level_enums = level_enums  # List of [N] or [B, N] tensors
        self.num_levels = len(level_enums)

    def to(self, device: torch.device, **kwargs):
        """Move all enumeration vectors to device.

        Args:
            device: Target device
            **kwargs: Additional arguments like non_blocking, passed to tensor.to()
        """
        self.level_enums = [e.to(device, **kwargs) for e in self.level_enums]
        return self


def create_hierarchical_mask(
    positions: HierarchicalPositions,
    window_size_per_level: list,
    max_level: int = None,
    batch_size: int = 1,
    num_heads: int = 8,
    device: torch.device = None,
):
    """
    Create vectorized cascading per-level sliding window masks.

    All levels checked simultaneously via:
    1. Stack enumeration vectors
    2. Vectorized participation + distance checks
    3. OR reduction (any level allows = can attend)

    Args:
        positions: HierarchicalPositions with sparse enumeration vectors
        window_size_per_level: [256, 64, 16] for tokens/sentences/sections
        max_level: Maximum active level (None = all levels)
        batch_size: batch size
        num_heads: number of attention heads
        device: torch device

    Returns:
        FlexAttention BlockMask with cascading OR-merged windows
    """
    device = device or positions.level_enums[0].device

    # Determine active levels
    num_active = (
        (max_level + 1) if max_level is not None else len(positions.level_enums)
    )

    # Stack enums: [Num_Levels, Seq_Len] or [Num_Levels, B, Seq_Len]
    stacked_enums = torch.stack(positions.level_enums[:num_active], dim=0).to(device)

    # Window sizes: [Num_Levels] tensor
    windows = torch.tensor(
        window_size_per_level[:num_active], device=device, dtype=torch.long
    )

    has_batch_dim = stacked_enums.ndim == 3
    seq_len = stacked_enums.shape[-1]

    def hierarchical_mask_fn(b, h, q_idx, kv_idx):
        """Vectorized: check all levels at once, OR-reduce.

        Args:
            b, h, q_idx, kv_idx: scalar indices
        Returns:
            scalar boolean indicating if query at q_idx can attend to key at kv_idx
        """
        # Fetch IDs for all levels: [Num_Levels]
        if has_batch_dim:
            q_ids = stacked_enums[:, b, q_idx]
            kv_ids = stacked_enums[:, b, kv_idx]
        else:
            q_ids = stacked_enums[:, q_idx]
            kv_ids = stacked_enums[:, kv_idx]

        # 1. Participation: both must be non-zero
        participates = (q_ids != 0) & (kv_ids != 0)

        # 2. Distance check: within window (enumeration space)
        dists = torch.abs(q_ids - kv_ids)
        in_window = dists <= windows

        # 3. Combine: participate AND in_window for each level
        valid_connections = participates & in_window

        # 4. OR reduction: any level allows = can attend
        return valid_connections.any()

    return create_block_mask(
        hierarchical_mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )


def _mlswa_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
) -> torch.Tensor:
    """Inner implementation of multi-level sliding window attention (compiled for performance)."""
    return flex_attention(query, key, value, block_mask=block_mask)


# Compile the flex_attention call for better performance
_compiled_mlswa_attention = torch.compile(_mlswa_attention_impl)


def mlswa_attention(
    query: torch.Tensor,  # [B, N, H, D]
    key: torch.Tensor,  # [B, N, H, D]
    value: torch.Tensor,  # [B, N, H, D]
    positions: HierarchicalPositions,
    window_size_per_level: list,
    max_level: int = None,
) -> torch.Tensor:
    """
    Cascading per-level sliding window attention (vectorized).

    All positions attend via level 0 window. Boundary positions (non-zero in
    level 1/2) additionally attend via their respective level windows (OR merge).

    Distance measured in enumeration space, not physical positions.

    Args:
        query, key, value: [B, N, H, D] attention inputs (batch, seq, heads, head_dim)
        positions: HierarchicalPositions with sparse enumeration vectors
        window_size_per_level: [256, 64, 16] = window sizes for tokens, sentences, sections
        max_level: Maximum hierarchy level allowed (None = all levels)

    Returns:
        [B, N, H, D] attention output
    """
    B, N, H, D = query.shape
    device = query.device

    # flex_attention expects [B, H, N, D] format, so we need to transpose
    query = query.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]
    key = key.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]
    value = value.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]

    block_mask = create_hierarchical_mask(
        positions=positions,
        window_size_per_level=window_size_per_level,
        max_level=max_level,
        batch_size=B,
        num_heads=H,
        device=device,
    )

    # Call compiled flex_attention with [B, H, N, D] tensors
    out = _compiled_mlswa_attention(query, key, value, block_mask)

    # Transpose back to [B, N, H, D] format
    out = out.transpose(1, 2)  # [B, H, N, D] -> [B, N, H, D]

    return out
