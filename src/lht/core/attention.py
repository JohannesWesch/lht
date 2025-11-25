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
    Memory-efficient lazy hierarchical mask (vmap-safe, scalable to 128k+ tokens).

    Uses loop unrolling to avoid advanced slicing on captured tensors,
    fixing the 'vmap ... .item()' error.

    Instead of pre-computing an [N, N] matrix (which would be 16GB for N=128k),
    this captures a list of [B, N] enumeration tensors and computes the boolean
    logic on-the-fly inside the FlexAttention kernel.

    Complexity:
    - Memory: O(L * N) - Linear scaling!
    - Compute: O(1) per attention pair (fused into kernel)

    For each query-key pair, checks all levels:
        - enum_q[ℓ] != 0 and enum_k[ℓ] != 0 (both participate)
        - |enum_q[ℓ] - enum_k[ℓ]| <= window_size[ℓ] (within window)
    Then OR across levels (cascading).

    Args:
        positions: HierarchicalPositions with sparse enumeration vectors
        window_size_per_level: [256, 64, 16] for tokens/sentences/sections
        max_level: Maximum active level (None = all levels)
        batch_size: Batch size for BlockMask
        num_heads: Number of attention heads
        device: Target device (GPU) for attention computation

    Returns:
        FlexAttention BlockMask with lazy evaluation
    """
    device = device or positions.level_enums[0].device

    # 1. Determine active levels
    num_active = (
        (max_level + 1) if max_level is not None else len(positions.level_enums)
    )

    # 2. Prepare enumeration tensors as a LIST of [B, N] tensors
    # We do NOT stack them into [L, B, N] to avoid vmap slicing issues.
    # Memory footprint: O(L * N) instead of O(N^2)!
    raw_enums = positions.level_enums[:num_active]
    processed_enums_list = []

    for e in raw_enums:
        e = e.to(device)  # Move to target device
        if e.ndim == 1:
            # [N] -> [1, N] -> [B, N]
            e = e.unsqueeze(0).expand(batch_size, -1)
        else:
            # [B, N] - ensure correct batch size
            e = e if e.shape[0] == batch_size else e.expand(batch_size, -1)
        processed_enums_list.append(e)

    seq_len = processed_enums_list[0].shape[-1]

    # Store window sizes for the closure (plain Python list for vmap safety)
    active_windows = window_size_per_level[:num_active]

    # 3. Lazy mask function with UNROLLED LOOP
    # This avoids `dense_enums[:, b, q_idx]` which breaks vmap
    def hierarchical_mask_fn(b, h, q_idx, kv_idx):
        """
        Iterates over levels explicitly (unrolled by compiler).
        Uses simple [B, N] indexing which is vmap-safe.
        """
        final_condition = None

        # Iterate over the list of tensors (unrolled by compiler for small L)
        for i, enum_tensor in enumerate(processed_enums_list):
            window = active_windows[i]

            # Standard indexing: [B, N][b, idx] - this is vmap-safe!
            q_val = enum_tensor[b, q_idx]
            k_val = enum_tensor[b, kv_idx]

            # Participation: both must be non-zero
            participates = (q_val != 0) & (k_val != 0)

            # Distance check in enumeration space
            dist = (q_val - k_val).abs()
            in_window = dist <= window

            # Valid for this level
            valid_level = participates & in_window

            # Accumulate via OR
            if final_condition is None:
                final_condition = valid_level
            else:
                final_condition = final_condition | valid_level

        return final_condition

    # 4. Create BlockMask with lazy evaluation
    # The mask function above is traced and fused into the attention kernel
    return create_block_mask(
        hierarchical_mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
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
    query = query.transpose(1, 2).contiguous()  # [B, N, H, D] -> [B, H, N, D]
    key = key.transpose(1, 2).contiguous()  # [B, N, H, D] -> [B, H, N, D]
    value = value.transpose(1, 2).contiguous()  # [B, N, H, D] -> [B, H, N, D]

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
    out = out.transpose(1, 2).contiguous()  # [B, H, N, D] -> [B, N, H, D]

    return out
