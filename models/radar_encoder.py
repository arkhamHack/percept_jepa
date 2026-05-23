"""
Radar Encoder
==============
Encodes sparse radar point clouds into a fixed set of latent tokens
suitable for transformer-based fusion with visual features.

Architecture
------------

Input: [B, N, 6]  — N radar points, each with 6 features
                    (x, y, z, rcs, vx_comp, vy_comp)

Step 1 — PointNet MLP backbone (per-point, shared weights)
  Each point is independently lifted to a high-dimensional feature via
  a stack of 1-D convolutions (equivalent to shared MLPs):
    [B, N, 6] → [B, N, 64] → [B, N, 128] → [B, N, C_max]

Step 2 — Global max pooling → global context vector [B, C_max]

Step 3 — Concatenate global context back to each point:
  [B, N, C_max + C_max] → [B, N, C_fused]

Step 4 — Soft token aggregation
  Instead of hard K-means clustering, we learn K soft assignment weights
  (attention scores) that aggregate N points into K radar tokens:
  [B, N, C_fused] → [B, K, D]
  This is differentiable and order-invariant (like Slot Attention lite).

Step 5 — Positional encoding
  Add 3-D spatial positional embedding to each token based on the weighted
  centroid of its assigned points.

Why this design?
  - Radar point clouds are sparse, unordered, and irregular — PointNet
    is the natural choice for permutation-invariant processing.
  - Aggregating to K fixed tokens is necessary for concatenation with
    the fixed visual token sequence in the fusion transformer.
  - Soft assignment is more expressive than hard clustering and avoids
    empty cluster problems.

TODO:
  - Replace soft attention aggregation with Slot Attention.
  - Add velocity-aware positional encoding (Doppler embedding).
  - Extend to 3-D occupancy grid projection.
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.positional_encoding import FourierPositionalEncoding3D

logger = logging.getLogger(__name__)


class PointNetBackbone(nn.Module):
    """
    Per-point PointNet-style MLP with global context concatenation.

    Processes each of the N points independently (shared MLP weights),
    then appends a global max-pooled context vector to each point feature.

    Input:  [B, N, input_dim]
    Output: [B, N, hidden_dims[-1] * 2]  (local + global concatenation)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.1,
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
            ]
            if dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    Tensor[B, N, input_dim]
            mask: Tensor[B, N]  (1 = real point, 0 = padding)

        Returns:
            Tensor[B, N, 2 * out_dim]  local+global feature per point
        """
        B, N, _ = x.shape

        # Per-point MLP: treat (B*N) as independent batch dim for BN
        x_flat = rearrange(x, 'b n d -> (b n) d')
        feat = self.mlp(x_flat)               # [(B*N), C]
        feat = rearrange(feat, '(b n) c -> b n c', b=B, n=N)
        # feat: [B, N, C]

        # Mask padding points before global pooling
        if mask is not None:
            # Set padding points to -inf for max pooling
            neg_inf_mask = (1 - mask.unsqueeze(-1)) * (-1e9)  # [B, N, 1]
            feat_masked = feat + neg_inf_mask
        else:
            feat_masked = feat

        # Global max pooling across N dimension
        global_feat = feat_masked.max(dim=1).values  # [B, C]
        global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, C]

        # Concatenate local + global features for each point
        combined = torch.cat([feat, global_feat], dim=-1)  # [B, N, 2*C]
        return combined


class SoftTokenAggregation(nn.Module):
    """
    Aggregate N variable radar points into K fixed latent tokens via
    learned soft attention weights.

    For each of K tokens, we compute attention scores over all N points,
    then take a weighted sum of point features.

    Input:  [B, N, in_dim]
    Output: [B, K, out_dim]

    This is inspired by:
      - DeepSets + attention pooling
      - Slot Attention (Locatello et al. 2020)
    """

    def __init__(self, in_dim: int, out_dim: int, num_tokens: int):
        super().__init__()
        self.K = num_tokens

        # K learnable query vectors
        self.queries = nn.Parameter(torch.randn(1, num_tokens, out_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.key_proj   = nn.Linear(in_dim, out_dim)
        self.value_proj = nn.Linear(in_dim, out_dim)
        self.scale = out_dim ** -0.5

        self.norm = nn.LayerNorm(out_dim)
        self.ff = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    Tensor[B, N, in_dim]
            mask: Tensor[B, N]  (1 = real, 0 = padding)

        Returns:
            Tensor[B, K, out_dim]
        """
        B, N, _ = x.shape

        K_vec = self.key_proj(x)    # [B, N, D]
        V_vec = self.value_proj(x)  # [B, N, D]

        Q = self.queries.expand(B, -1, -1)  # [B, K, D]

        # Attention: [B, K, N]
        attn = torch.bmm(Q, K_vec.transpose(1, 2)) * self.scale

        # Mask padding points: set their attention logits to -inf
        if mask is not None:
            attn_mask = (1 - mask).bool().unsqueeze(1)  # [B, 1, N]
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)  # [B, K, N]

        # Handle all-padding edge case (softmax of all -inf → nan)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum of values
        tokens = torch.bmm(attn, V_vec)  # [B, K, D]

        # Residual + LayerNorm + FFN
        tokens = self.norm(tokens + Q)
        tokens = self.norm2(tokens + self.ff(tokens))

        return tokens  # [B, K, D]


class RadarEncoder(nn.Module):
    """
    Complete radar point cloud encoder.

    Transforms [B, N, 6] sparse radar points into
    [B, K, D] dense latent radar tokens for fusion.

    Args:
        cfg: full config (cfg.model.radar sub-section expected)
    """

    def __init__(self, cfg):
        super().__init__()
        rcfg = cfg.model.radar
        self.hidden_dim = cfg.model.hidden_dim
        self.num_tokens = rcfg.num_radar_tokens   # K

        input_dim   = rcfg.input_dim              # 6
        hidden_dims = list(rcfg.pointnet_hidden)  # e.g. [64, 128, 256]
        dropout     = rcfg.dropout

        # Step 1-2: PointNet per-point encoding
        self.pointnet = PointNetBackbone(input_dim, hidden_dims, dropout)
        pointnet_out_dim = hidden_dims[-1] * 2  # local + global concat

        # Optional: project to hidden_dim before aggregation
        self.pre_agg_proj = nn.Linear(pointnet_out_dim, self.hidden_dim)

        # Step 3: Soft aggregation → K tokens
        self.aggregation = SoftTokenAggregation(
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            num_tokens=self.num_tokens,
        )

        # Step 4: Spatial positional encoding (added after aggregation)
        # We use a simple 3-D Fourier positional encoding based on the
        # *weighted centroid* of each token's attention weights.
        self.pos_enc = FourierPositionalEncoding3D(
            d_model=self.hidden_dim,
            max_freq=8,
        )

        # Final layer norm
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        radar_pts: torch.Tensor,     # [B, N, 6]
        radar_mask: torch.Tensor,    # [B, N]   1=real, 0=pad
    ) -> torch.Tensor:
        """
        Encode radar point cloud to latent tokens.

        Args:
            radar_pts:  Tensor[B, N, 6]   (x, y, z, rcs, vx_comp, vy_comp)
            radar_mask: Tensor[B, N]       1 = real point

        Returns:
            Tensor[B, K, D_hidden]
              K = num_radar_tokens (e.g. 64)
              D_hidden = hidden_dim (e.g. 768)

        Each of the K output tokens represents a learned aggregation
        of the radar scatter pattern.  The aggregation is differentiable
        and order-invariant thanks to the PointNet + soft attention design.
        """
        B, N, _ = radar_pts.shape

        # ---- Step 1-2: PointNet backbone (per-point local+global feats) --
        pointnet_feats = self.pointnet(radar_pts, radar_mask)  # [B, N, 2*C]

        # ---- Project to hidden_dim -----------------------------------
        pointnet_feats = self.pre_agg_proj(pointnet_feats)     # [B, N, D]

        # ---- Step 3: Soft token aggregation -------------------------
        tokens = self.aggregation(pointnet_feats, radar_mask)   # [B, K, D]

        # ---- Step 4: Spatial positional encoding --------------------
        # Compute the (x, y, z) centroid of real radar points for each
        # sample (used as a coarse positional context for the K tokens)
        xyz = radar_pts[..., :3]      # [B, N, 3]
        valid = radar_mask.unsqueeze(-1).float()  # [B, N, 1]
        n_valid = valid.sum(dim=1).clamp(min=1)   # [B, 1]
        centroid = (xyz * valid).sum(dim=1) / n_valid   # [B, 3]

        # Expand centroid across K tokens (all tokens share same origin;
        # TODO: use per-token attention centroid for richer positional info)
        centroid_tokens = centroid.unsqueeze(1).expand(-1, self.num_tokens, -1)
        # centroid_tokens: [B, K, 3]

        pe = self.pos_enc(centroid_tokens)  # [B, K, D]
        tokens = tokens + pe

        tokens = self.output_norm(tokens)   # [B, K, D]
        return tokens
