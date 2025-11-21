"""
HierarchyManager for variable-depth learned hierarchies.

This module provides a unified interface for managing K abstraction levels:
- 2 levels: tokens → sentences
- 3 levels: tokens → sentences → sections (default)
- 4 levels: tokens → sentences → paragraphs → sections
- N levels: arbitrary depth

The HierarchyManager:
- Creates K routers (one per abstraction level)
- Runs them sequentially during forward pass
- Maintains level_id[ℓ] and is_head[ℓ] for each level ℓ = 1..K
- Computes ratio losses for all levels
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .routers import LevelRouter


class HierarchyManager(nn.Module):
    """
    Manages a variable-depth learned hierarchy with K abstraction levels.

    Each level ℓ (where ℓ = 1..K) has:
    - A router that predicts which tokens/units are "heads" at that level
    - level_id[ℓ]: tensor mapping each token to its parent unit at level ℓ
    - is_head[ℓ]: binary tensor indicating which tokens are heads at level ℓ

    Example with K=3 (tokens → sentences → sections):
    - Level 1: sentence boundaries (token → sentence)
    - Level 2: section boundaries (sentence → section)
    - Level 3 would add another abstraction level

    Example with K=2 (tokens → sentences only):
    - Level 1: sentence boundaries
    """

    def __init__(self, hierarchy_config, d_model: int):
        super().__init__()
        self.cfg = hierarchy_config
        self.num_levels = len(hierarchy_config.levels)  # K
        self.d_model = d_model

        # Create K routers, one for each abstraction level
        self.routers = nn.ModuleList(
            [
                LevelRouter(
                    level_name=level_cfg.name,
                    d_model=d_model,
                    router_config=hierarchy_config.router,
                )
                for level_cfg in hierarchy_config.levels
            ]
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run all K routers in sequence to build the full hierarchy.

        NOTE: This method runs all routers at once. For schedule-driven
        routing (routers at different layer depths), use update_single_level().

        Args:
            hidden: [B, N, D] token representations (after initial layers)
            attention_mask: [B, N] mask for valid tokens

        Returns:
            dict with:
                - level_ids: Dict mapping level -> [B, N] tensor
                - is_heads: Dict mapping level -> [B, N] tensor
                - router_outputs: List of raw router outputs for loss computation
        """
        _batch_size, _seq_len, _ = hidden.shape
        _device = hidden.device

        level_ids = {}  # level -> [B, N]
        is_heads = {}  # level -> [B, N]
        router_outputs = []

        # Run routers sequentially
        for level_idx, router in enumerate(self.routers):
            level_num = level_idx + 1  # 1-indexed

            if level_idx == 0:
                router_out = router(
                    hidden=hidden,
                    attention_mask=attention_mask,
                )
            else:
                router_out = router(
                    hidden=hidden,
                    prev_level_ids=level_ids[level_idx],
                    prev_is_heads=is_heads[level_idx],
                    attention_mask=attention_mask,
                )

            level_ids[level_num] = router_out["level_ids"]
            is_heads[level_num] = router_out["is_head"]
            router_outputs.append(router_out)

        return {
            "level_ids": level_ids,  # Dict[int, Tensor[B, N]]
            "is_heads": is_heads,  # Dict[int, Tensor[B, N]]
            "router_outputs": router_outputs,
            "num_levels": self.num_levels,
        }

    def update_single_level(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        target_level: int,
        state: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Run router for a single level (schedule-driven).

        Args:
            hidden: [B, N, D] current representations
            attention_mask: [B, N] mask for valid tokens
            target_level: which level to compute (1-indexed)
            state: current hierarchy state with level_ids/is_heads/ratio_losses

        Returns:
            updated state dict
        """
        if "level_ids" not in state:
            state["level_ids"] = {}
            state["is_heads"] = {}
            state["ratio_losses"] = {}

        level_idx = target_level - 1
        lvl_cfg = self.cfg.levels[level_idx]
        router = self.routers[level_idx]

        # Run router
        if target_level == 1:
            router_out = router(hidden, attention_mask=attention_mask)
        else:
            # Use previous level's info
            prev_level = target_level - 1
            router_out = router(
                hidden=hidden,
                prev_level_ids=state["level_ids"].get(prev_level),
                prev_is_heads=state["is_heads"].get(prev_level),
                attention_mask=attention_mask,
            )

        # Update state
        state["level_ids"][target_level] = router_out["level_ids"]
        state["is_heads"][target_level] = router_out["is_head"]

        # Compute ratio loss
        probs = router_out["probs"]
        mask = attention_mask.float()
        num_valid = mask.sum(dim=-1).clamp(min=1.0)
        expected_heads = (probs * mask).sum(dim=-1) / num_valid
        ratio_loss = (expected_heads - lvl_cfg.target_head_ratio).abs().mean()
        weighted_loss = lvl_cfg.loss_weight * ratio_loss

        state["ratio_losses"][target_level] = weighted_loss

        return state

    def compute_losses(
        self,
        hierarchy_output: Dict,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ratio losses for all K levels.

        Args:
            hierarchy_output: output from forward()
            attention_mask: [B, N] mask for valid tokens

        Returns:
            dict with individual and total losses
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(hierarchy_output["level_ids"][0])

        losses = {}
        total_loss = 0.0

        for level_idx, (level_cfg, router_out) in enumerate(
            zip(self.cfg.levels, hierarchy_output["router_outputs"])
        ):
            # Compute ratio loss for this level
            probs = router_out["probs"]
            target_ratio = level_cfg.target_head_ratio
            weight = level_cfg.loss_weight

            num_valid = attention_mask.sum(dim=-1).clamp(min=1.0)
            expected_heads = (probs * attention_mask).sum(dim=-1) / num_valid
            ratio_loss = (expected_heads - target_ratio).abs().mean()
            weighted_loss = weight * ratio_loss

            level_name = level_cfg.name
            losses[f"ratio_loss_{level_name}"] = ratio_loss
            losses[f"weighted_loss_{level_name}"] = weighted_loss
            total_loss = total_loss + weighted_loss

        losses["total_hierarchy_loss"] = total_loss
        return losses

    def get_statistics(
        self,
        hierarchy_output: Dict,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute interpretable statistics about the learned hierarchy.

        Args:
            hierarchy_output: output from forward()
            attention_mask: [B, N] mask for valid tokens

        Returns:
            dict with compression ratios, avg units per doc, etc.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(hierarchy_output["level_ids"][0])

        stats = {}
        num_valid = attention_mask.sum()

        for level_idx, (level_cfg, is_head) in enumerate(
            zip(self.cfg.levels, hierarchy_output["is_heads"])
        ):
            level_name = level_cfg.name
            num_heads = (is_head * attention_mask).sum()

            stats[f"avg_{level_name}s_per_doc"] = (num_heads / num_valid).item()
            stats[f"compression_{level_name}"] = (
                num_valid / num_heads.clamp(min=1)
            ).item()

        return stats
