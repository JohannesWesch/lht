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
    radius: int,
    batch_size: int = 1,
    num_heads: int = 8,
    device: torch.device = None,
):
    """
    Create FlexAttention mask: attend if |Δx| + |Δy| ≤ radius.

    Args:
        coords: GeometricCoordinates
        radius: Manhattan distance threshold
        batch_size: batch size
        num_heads: number of heads
        device: device

    Returns:
        FlexAttention BlockMask
    """
    device = device or coords.levels.device
    coords = coords.to(device)
    seq_len = len(coords.levels)

    levels = coords.levels.to(device)
    logical_times = coords.logical_times.to(device)

    def geometric_mask_fn(b, h, q_idx, kv_idx):
        """Mask function: True if distance ≤ radius."""
        q_level = levels[q_idx]
        k_level = levels[kv_idx]
        q_time = logical_times[q_idx]
        k_time = logical_times[kv_idx]

        # Manhattan distance
        dist_level = (q_level[:, None] - k_level[None, :]).abs()
        dist_time = (q_time[:, None] - k_time[None, :]).abs()
        manhattan_dist = dist_level + dist_time

        return manhattan_dist <= radius

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
    radius: int = 2,
) -> torch.Tensor:
    """
    Geometric attention with Manhattan distance sliding window.

    Args:
        query, key, value: [B, N, H, D] attention inputs
        coords: GeometricCoordinates
        radius: Manhattan distance threshold

    Returns:
        [B, N, H, D] attention output
    """
    B, N, H, D = query.shape
    device = query.device

    block_mask = create_geometric_mask(
        coords=coords,
        radius=radius,
        batch_size=B,
        num_heads=H,
        device=device,
    )

    return flex_attention(query, key, value, block_mask=block_mask)
