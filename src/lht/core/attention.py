"""
Core geometric attention mechanism.

Implements Manhattan distance sliding window attention:
    attend if |Δx| + |Δy| ≤ R
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class GeometricCoordinates:
    """
    2D coordinates for hierarchical sequences.

    Each token gets (x, y) where:
        - y = hierarchy_level (0=token, 1=sentence, 2=paragraph, ...)
        - x = logical_time (parent group ID)
    """

    def __init__(
        self,
        levels: torch.Tensor,  # [N] or [B, N] - hierarchy level
        logical_times: torch.Tensor,  # [N] or [B, N] - parent group ID
        physical_positions: torch.Tensor = None,  # [N] or [B, N] - optional
    ):
        self.levels = levels
        self.logical_times = logical_times
        self.physical_positions = (
            physical_positions
            if physical_positions is not None
            else torch.arange(
                levels.shape[-1], dtype=torch.long, device=levels.device
            ).expand_as(levels)
        )

    def to(self, device: torch.device, **kwargs):
        """Move to device.

        Args:
            device: Target device
            **kwargs: Additional arguments like non_blocking, passed to tensor.to()
        """
        self.levels = self.levels.to(device, **kwargs)
        self.logical_times = self.logical_times.to(device, **kwargs)
        self.physical_positions = self.physical_positions.to(device, **kwargs)
        return self


def create_geometric_mask(
    coords: GeometricCoordinates,
    radius: int = 1,
    max_level: int = None,
    window_size_per_level: list = None,
    batch_size: int = 1,
    num_heads: int = 8,
    device: torch.device = None,
):
    """
    Create FlexAttention mask with geometric + sliding window constraints.

    Args:
        coords: GeometricCoordinates
        radius: Manhattan distance threshold (typically 1 for parent-child)
        max_level: Maximum hierarchy level allowed in this layer (None = all levels)
        window_size_per_level: [512, 64, 16] = max elements per level (tokens, sentences, sections)
        batch_size: batch size
        num_heads: number of heads
        device: device

    Returns:
        FlexAttention BlockMask
    """
    device = device or coords.levels.device
    coords = coords.to(device)

    # Support [B, N] or [N] coords
    # If [N], we rely on broadcasting or assume it's same for all batch
    # If [B, N], we must index with 'b'

    has_batch_dim = coords.levels.ndim == 2
    seq_len = coords.levels.shape[-1]

    levels = coords.levels.to(device)
    logical_times = coords.logical_times.to(device)

    # Convert window sizes to tensor if provided
    if window_size_per_level is not None:
        window_sizes = torch.tensor(
            window_size_per_level, device=device, dtype=torch.long
        )
    else:
        window_sizes = None

    def geometric_mask_fn(b, h, q_idx, kv_idx):
        """Mask function: geometric + sliding window constraints.

        Args:
            b, h, q_idx, kv_idx: scalar indices
        Returns:
            scalar boolean indicating if query at q_idx can attend to key at kv_idx
        """
        # Get scalar values for this query-key pair
        if has_batch_dim:
            q_level = levels[b, q_idx]
            k_level = levels[b, kv_idx]
            q_time = logical_times[b, q_idx]
            k_time = logical_times[b, kv_idx]
        else:
            q_level = levels[q_idx]
            k_level = levels[kv_idx]
            q_time = logical_times[q_idx]
            k_time = logical_times[kv_idx]

        # Manhattan distance (for parent-child connectivity)
        dist_level = torch.abs(q_level - k_level)
        dist_time = torch.abs(q_time - k_time)
        manhattan_dist = dist_level + dist_time

        within_radius = manhattan_dist <= radius

        # Per-layer active levels control
        if max_level is not None:
            level_ok = (q_level <= max_level) & (k_level <= max_level)
            within_radius = within_radius & level_ok

        # Sliding window constraint per level
        if window_sizes is not None:
            # For same-level pairs, apply window size constraint
            same_level = q_level == k_level

            # Check if within window for this level
            # Use torch.where to avoid data-dependent control flow
            within_window = dist_time <= window_sizes[q_level]
            # Apply window constraint only for same-level pairs
            # Logic: keep if (not same level) OR (same level AND within window)
            within_radius = within_radius & (~same_level | within_window)

        return within_radius

    return create_block_mask(
        geometric_mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )


def _geometric_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
) -> torch.Tensor:
    """Inner implementation of geometric attention (compiled for performance)."""
    return flex_attention(query, key, value, block_mask=block_mask)


# Compile the flex_attention call for better performance
_compiled_flex_attention = torch.compile(_geometric_attention_impl)


def geometric_attention(
    query: torch.Tensor,  # [B, N, H, D]
    key: torch.Tensor,  # [B, N, H, D]
    value: torch.Tensor,  # [B, N, H, D]
    coords: GeometricCoordinates,
    radius: int = 1,
    max_level: int = None,
    window_size_per_level: list = None,
) -> torch.Tensor:
    """
    Geometric attention with Manhattan distance + sliding window.

    Args:
        query, key, value: [B, N, H, D] attention inputs (batch, seq, heads, head_dim)
        coords: GeometricCoordinates
        radius: Manhattan distance threshold (typically 1 for parent-child)
        max_level: Maximum hierarchy level allowed (None = all levels)
        window_size_per_level: [512, 64, 16] = window sizes for tokens, sentences, sections

    Returns:
        [B, N, H, D] attention output
    """
    B, N, H, D = query.shape
    device = query.device

    # flex_attention expects [B, H, N, D] format, so we need to transpose
    query = query.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]
    key = key.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]
    value = value.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]

    block_mask = create_geometric_mask(
        coords=coords,
        radius=radius,
        max_level=max_level,
        window_size_per_level=window_size_per_level,
        batch_size=B,
        num_heads=H,
        device=device,
    )

    # Call compiled flex_attention with [B, H, N, D] tensors
    out = _compiled_flex_attention(query, key, value, block_mask)

    # Transpose back to [B, N, H, D] format
    out = out.transpose(1, 2)  # [B, H, N, D] -> [B, N, H, D]

    return out
