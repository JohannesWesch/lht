"""
Core coordinate builder - universal only.

Single function to build coordinates for any L-level hierarchy.
"""

import torch

from .attention import GeometricCoordinates


def build_coords(
    parent_maps: list[list[int]],  # parent_maps[ℓ][i] = parent ID of node i at level ℓ
    device: torch.device = None,
) -> GeometricCoordinates:
    """
    Build coordinates for arbitrary L-level hierarchy.

    Core principle: logical_time[node at level ℓ] = parent_id at level ℓ+1
    This ensures Manhattan distance child→parent = 1

    Key insight: All children of the same parent share the same x-coordinate.
    The parent summary is at that same x, but y+1.

    Args:
        parent_maps: List of parent ID lists, one per level
            parent_maps[0][i] = parent_id (sentence_id) for token i
            parent_maps[1][i] = parent_id (paragraph_id) for sentence i
            ...
            parent_maps[L-1][i] = own_id for top-level node i
        device: torch device

    Returns:
        GeometricCoordinates with guaranteed distance = 1 for all parent-child pairs

    Example (3 levels):
        parent_maps = [
            [0, 0, 0, 1, 1],  # 5 tokens: 3 in sent 0, 2 in sent 1
            [0, 0],            # 2 sentences: both in para 0
            [0],               # 1 paragraph: top level (own ID)
        ]
        Results in:
            Tokens 0,1,2 → (x=0, y=0)  all same x = sentence_id
            Tokens 3,4   → (x=1, y=0)  all same x = sentence_id
            Sentence 0   → (x=0, y=1)  same x as its tokens
            Sentence 1   → (x=1, y=1)  same x as its tokens
            Paragraph 0  → (x=0, y=2)  same x as its sentences

        Then: token ↔ sentence parent → |Δx|=0, |Δy|=1 → distance=1 ✅
    """
    device = device or torch.device("cpu")

    levels = []
    logical_times = []

    for level_idx, parent_list in enumerate(parent_maps):
        for node_id, parent_id in enumerate(parent_list):
            levels.append(level_idx)
            logical_times.append(parent_id)

    return GeometricCoordinates(
        levels=torch.tensor(levels, dtype=torch.long, device=device),
        logical_times=torch.tensor(logical_times, dtype=torch.long, device=device),
        physical_positions=torch.arange(len(levels), dtype=torch.long, device=device),
    )
