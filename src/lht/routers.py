"""
H-Net style dynamic chunking router.

This module implements the dynamic chunking mechanism from the H-Net paper
(https://arxiv.org/abs/2507.07955):
- Cosine similarity between adjacent projected states (Eq. 4)
- Boundary probabilities: p_t = 0.5 * (1 - cos(q_t, k_{t-1}))
- Hard boundary indicators: b_t = 1[p_t >= 0.5]
- Ratio loss encouraging target compression factor (Eq. 10)
- No Gumbel, no stochastic routing - just cosine similarity and hard thresholding

NOTE: H-Net's upsampling module is NOT implemented here because LHT is a
hierarchical transformer (progressive abstraction), not an autoencoder
(compression → reconstruction). See LevelRouter docstring for full rationale.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelRouter(nn.Module):
    """
    H-Net style routing module (Dynamic Chunking) for a single hierarchy level.

    Implements equation (4) and ratio loss from equation (10) of the paper:
      - Cosine similarity between adjacent projected states
      - Boundary probability p_t = 0.5 * (1 - cos(q_t, k_{t-1}))
      - Hard boundary indicator b_t = 1[p_t >= 0.5], with p_1 := 1
      - Group ids via cumulative sum of b_t
      - Ratio loss encouraging expected compression factor N

    No Gumbel, no stochastic routing. Gradients flow through p_t and the ratio
    loss, just like in H-Net.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🔥 IMPORTANT: Why LHT Does NOT Need H-Net's Upsampling Module
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    H-Net is an autoencoder:
      Encoder compresses → Decoder reconstructs → Must upsample chunks back to token length

    LHT is a hierarchical transformer:
      Tokens → Sentences → Sections → Document (progressive abstraction, no reconstruction)

    Key differences:
      1. H-Net needs to reconstruct full-resolution sequences (upsampling required)
      2. LHT never reverses the abstraction (upsampling would break hierarchy)
      3. LHT's higher layers INTENTIONALLY don't need token-level resolution
      4. Upsampling would reintroduce O(N²) attention cost, defeating the purpose

    Therefore, LHT uses:
      ✅ Dynamic chunk boundary detection (Eq. 4)
      ✅ Ratio loss (Eq. 10)
      ❌ Smoothing module (autoencoder-specific)
      ❌ Upsampling module (autoencoder-specific)
      ❌ Dechunking STE (autoencoder-specific)

    This is a deliberate architectural decision, not an omission.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(
        self,
        level_name: str,
        d_model: int,
        target_head_ratio: float,  # e.g. 0.2 -> target N≈5
        loss_weight: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.level_name = level_name
        self.d_model = d_model
        self.target_head_ratio = target_head_ratio
        self.loss_weight = loss_weight
        self.eps = eps

        # In H-Net they use separate projections W_q, W_k for similarity.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

    # ---- routing math ----

    def _cosine_adjacent(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between consecutive projected states.

        hidden: [B, N, D]
        returns sim: [B, N] where sim[:, t] ~ cos(q_t, k_{t-1})
        (we pad t=0 with the value at t=1 just for shape convenience)
        """
        B, N, D = hidden.shape

        q = self.W_q(hidden)  # [B, N, D]
        k = self.W_k(hidden)  # [B, N, D]

        # cosine between t and t-1 for t >= 1
        q_t = q[:, 1:, :]  # [B, N-1, D]
        k_prev = k[:, :-1, :]  # [B, N-1, D]

        # normalize
        q_n = F.normalize(q_t, p=2, dim=-1)
        k_n = F.normalize(k_prev, p=2, dim=-1)

        sim = (q_n * k_n).sum(dim=-1)  # [B, N-1]

        # pad first position to keep [B, N] (won't matter because we'll set p_0=1)
        first = sim[:, :1].detach()
        sim = torch.cat([first, sim], dim=1)  # [B, N]
        return sim

    def _boundary_probs(
        self,
        sim: torch.Tensor,  # [B, N]
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Equation (4): p_t = 0.5 * (1 - cos(q_t, k_{t-1})) with p_1 = 1.0.
        """
        # map cos ∈ [-1,1] -> p ∈ [0,1]
        p = 0.5 * (1.0 - sim)  # [B, N]

        if attention_mask is not None:
            p = p * attention_mask

        # first position is always start of a chunk
        p[:, 0] = 1.0
        return p.clamp(0.0, 1.0)

    def _boundary_indicators(self, p: torch.Tensor) -> torch.Tensor:
        """
        Hard boundaries b_t = 1[p_t >= 0.5].
        There is *no* STE in the paper at this point — STE is used later
        in dechunking (eq. 7), not for boundary decisions.
        """
        b = (p >= 0.5).float()
        return b

    def _group_ids(
        self,
        b: torch.Tensor,  # [B, N]
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Cumulative sum of boundary indicators gives chunk IDs.
        """
        ids = torch.cumsum(b, dim=-1) - 1  # [-1,0,0,1,...] pattern

        if attention_mask is not None:
            ids = ids * attention_mask + (-1) * (1 - attention_mask)

        return ids.long()

    # ---- ratio loss (eq. 10) ----

    def _ratio_loss(
        self,
        b: torch.Tensor,  # [B, N], hard boundaries
        p: torch.Tensor,  # [B, N], soft probs
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Implements the ratio loss L_ratio from Eq. (10).

        F = fraction of actual selected positions (using b_t)
        G = average boundary probability

        Target: compression factor N ≈ 1 / target_head_ratio
        """
        if attention_mask is not None:
            mask = attention_mask.float()
            b = b * mask
            p = p * mask
            L = mask.sum(dim=-1).clamp(min=1.0)  # [B]
        else:
            L = torch.full((b.size(0),), b.size(1), device=b.device, dtype=torch.float)

        # per-batch fractions
        F = b.sum(dim=-1) / L  # [B]
        G = p.sum(dim=-1) / L  # [B]

        # target downsampling factor N
        # e.g. target_head_ratio=0.2 => N=5
        N = max(self.target_head_ratio, 1e-6) ** -1

        # Eq (10):
        # L_ratio = N/(N-1) * ( (N-1)*F*G + (1-F)*(1-G) )
        # Implemented elementwise then averaged.
        num = (N - 1.0) * F * G + (1.0 - F) * (1.0 - G)
        L_ratio = (N / (N - 1.0)) * num
        L_ratio = L_ratio.mean()

        return self.loss_weight * L_ratio

    # ---- public API ----

    def forward(
        self,
        hidden: torch.Tensor,  # [B, N, D]
        prev_level_ids: Optional[torch.Tensor] = None,  # unused for now
        prev_is_heads: Optional[torch.Tensor] = None,  # unused for now
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
          probs:       [B, N]  boundary probabilities p_t
          boundaries:  [B, N]  hard indicators b_t
          level_ids:   [B, N]  group ids per token
          ratio_loss:  scalar  loss to be added to training objective
        """
        # 1) cosine similarity between adjacent tokens
        sim = self._cosine_adjacent(hidden)  # [B, N]

        # 2) boundary probabilities p_t
        probs = self._boundary_probs(sim, attention_mask)

        # 3) hard boundary indicators b_t
        boundaries = self._boundary_indicators(probs)

        if attention_mask is not None:
            boundaries = boundaries * attention_mask

        # 4) group ids via cumulative sum
        level_ids = self._group_ids(boundaries, attention_mask)

        # 5) ratio loss (encourages target compression)
        ratio_loss = self._ratio_loss(boundaries, probs, attention_mask)

        return {
            "probs": probs,
            "boundaries": boundaries,
            "level_ids": level_ids,
            "ratio_loss": ratio_loss,
            "is_head": boundaries,  # alias for backward compatibility
        }
