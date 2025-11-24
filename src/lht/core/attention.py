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
        levels: torch.Tensor,  # [N] - hierarchy level
        logical_times: torch.Tensor,  # [N] - parent group ID
        physical_positions: torch.Tensor = None,  # [N] - optional
    ):
        self.levels = levels
        self.logical_times = logical_times
        self.physical_positions = (
            physical_positions
            if physical_positions is not None
            else torch.arange(len(levels), dtype=torch.long, device=levels.device)
        )

    def to(self, device: torch.device):
        """Move to device."""
        self.levels = self.levels.to(device)
        self.logical_times = self.logical_times.to(device)
        self.physical_positions = self.physical_positions.to(device)
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

    Example:
        max_level=1, window_size_per_level=[512, 64]
        → tokens can attend within 512 token window
        → sentences can attend within 64 sentence window
        → tokens ↔ sentences via Manhattan distance (parent-child)
    """
    device = device or coords.levels.device
    coords = coords.to(device)
    seq_len = len(coords.levels)

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
        """Mask function: geometric + sliding window constraints."""
        q_level = levels[q_idx]
        k_level = levels[kv_idx]
        q_time = logical_times[q_idx]
        k_time = logical_times[kv_idx]

        # Manhattan distance (for parent-child connectivity)
        dist_level = (q_level[:, None] - k_level[None, :]).abs()
        dist_time = (q_time[:, None] - k_time[None, :]).abs()
        manhattan_dist = dist_level + dist_time

        within_radius = manhattan_dist <= radius

        # Per-layer active levels control
        if max_level is not None:
            level_ok = (q_level[:, None] <= max_level) & (k_level[None, :] <= max_level)
            within_radius = within_radius & level_ok

        # Sliding window constraint per level
        if window_sizes is not None:
            # For same-level pairs, apply window size constraint
            same_level = q_level[:, None] == k_level[None, :]

            # For each level, check if logical_time diff is within window
            within_window = torch.ones_like(within_radius, dtype=torch.bool)
            for level_idx in range(len(window_sizes)):
                is_this_level = (q_level[:, None] == level_idx) & (
                    k_level[None, :] == level_idx
                )
                time_diff = dist_time
                level_window_ok = time_diff <= window_sizes[level_idx]
                # Update: for this level, must be within window
                within_window = torch.where(
                    is_this_level, level_window_ok, within_window
                )

            # Apply window constraint only for same-level pairs
            within_radius = within_radius & (~same_level | within_window)

        return within_radius

    return create_block_mask(
        geometric_mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )


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
        query, key, value: [B, N, H, D] attention inputs
        coords: GeometricCoordinates
        radius: Manhattan distance threshold (typically 1 for parent-child)
        max_level: Maximum hierarchy level allowed (None = all levels)
        window_size_per_level: [512, 64, 16] = window sizes for tokens, sentences, sections

    Returns:
        [B, N, H, D] attention output
    """
    B, N, H, D = query.shape
    device = query.device

    block_mask = create_geometric_mask(
        coords=coords,
        radius=radius,
        max_level=max_level,
        window_size_per_level=window_size_per_level,
        batch_size=B,
        num_heads=H,
        device=device,
    )

    return flex_attention(query, key, value, block_mask=block_mask)
