"""
Detection Head — DETR-style Object Query Decoder
==================================================
Implements a transformer decoder that attends over the shared scene latent
to produce class predictions and 3-D bounding boxes.

DETR (End-to-End Object Detection with Transformers, Carion et al. 2020)
philosophy:
  - N learnable "object queries" each represent a potential detected object.
  - The decoder cross-attends to scene latent tokens (memory) to fill each
    query with object-specific information.
  - No NMS required; queries compete via set-prediction + Hungarian matching.

Why object queries instead of anchor boxes?
  - Queries are not tied to any specific spatial location or scale.
  - The transformer can learn arbitrary spatial reasoning.
  - Much more natural for a latent-representation-based architecture.
  - Cleaner end-to-end training with set-based Hungarian loss.

How the queries attend to the scene latent:
  - Each query Q_i (D-dimensional) acts as a content query.
  - The decoder cross-attends Q_i over all V+K+1 scene tokens.
  - The attention pattern reveals *which* spatial patches and radar clusters
    are relevant for object Q_i.
  - This attention can be visualised to understand model reasoning.

Output per query:
  - class_logits:  [B, Q, num_classes + 1]  (+ 1 for "no object" / background)
  - pred_boxes:    [B, Q, 7]                (cx, cy, cz, dx, dy, dz, sin_yaw, cos_yaw)
                    NOTE: yaw encoded as (sin, cos) for smooth regression;
                    final output is 8-D but 7-D box is recovered by atan2.

TODO:
  - Add reference point refinement (Deformable DETR style).
  - Two-stage detection: first generate coarse proposals, then refine.
  - Uncertainty estimation heads (Bayesian detection).
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.shared_scene_latent import SceneLatent

logger = logging.getLogger(__name__)

# Standard nuScenes class labels + background
NUM_NUSCENES_CLASSES = 10
BG_CLASS_IDX = NUM_NUSCENES_CLASSES  # index 10 = "no object"


class DecoderLayer(nn.Module):
    """
    Standard transformer decoder layer:
      1. Self-attention among object queries (allows queries to differentiate)
      2. Cross-attention from queries to scene latent (fills queries with scene info)
      3. Feed-forward network

    All with pre-norm and residual connections.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,          # [B, Q, D]  object queries
        memory:  torch.Tensor,          # [B, N, D]  scene latent tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            queries:  updated [B, Q, D]
            attn_w:   cross-attention weights [B, Q, N]  (for visualisation)
        """
        # ---- 1. Self-attention among queries ------------------------
        residual = queries
        queries = self.norm1(queries)
        queries, _ = self.self_attn(queries, queries, queries)
        queries = residual + self.drop(queries)

        # ---- 2. Cross-attention: queries read scene latent ----------
        residual = queries
        queries_n = self.norm2(queries)
        updated, attn_w = self.cross_attn(queries_n, memory, memory)
        # attn_w: [B, Q, N]  — which scene tokens each query attends to
        queries = residual + self.drop(updated)

        # ---- 3. FFN -------------------------------------------------
        residual = queries
        queries = residual + self.drop(self.ff(self.norm3(queries)))

        return queries, attn_w


class DetectionHead(nn.Module):
    """
    DETR-style detection head.

    Learns Q object queries that decode into class predictions and 3-D boxes
    by attending over the shared scene latent.

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        dcfg = cfg.model.detection
        D     = cfg.model.hidden_dim
        Q     = dcfg.num_queries
        C     = dcfg.num_classes      # foreground classes
        L     = dcfg.decoder_layers
        H     = dcfg.decoder_heads
        drop  = dcfg.dropout
        ffn   = D * 4

        self.num_queries  = Q
        self.num_classes  = C

        # ---- Learnable object queries --------------------------------
        # Each query is a D-dimensional vector initialised randomly.
        # During forward pass these are broadcast over the batch.
        # The model learns to specialise each query for certain object types
        # or spatial locations through training.
        self.object_queries = nn.Parameter(torch.randn(1, Q, D))
        nn.init.trunc_normal_(self.object_queries, std=0.02)

        # ---- Transformer decoder layers -----------------------------
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(D, H, ffn, drop) for _ in range(L)
        ])
        self.final_norm = nn.LayerNorm(D)

        # ---- Prediction MLPs ----------------------------------------
        # Classification: Q → (C + 1) class logits  (+1 for background)
        self.class_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, C + 1),  # C foreground + 1 background
        )

        # Box regression: Q → 8 values
        # Encoding: (cx, cy, cz, log_dx, log_dy, log_dz, sin_yaw, cos_yaw)
        # We regress log(size) to ensure positive dimensions.
        self.box_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 8),
        )

        # Confidence score (auxiliary output, independent of class)
        self.conf_head = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 1),
        )

    def forward(
        self, scene: SceneLatent
    ) -> Dict[str, torch.Tensor]:
        """
        Decode object detections from the shared scene latent.

        Args:
            scene: SceneLatent with .tokens of shape [B, N_total, D]

        Returns:
            dict with:
              'class_logits':  Tensor[B, Q, C+1]   raw class logits
              'pred_boxes':    Tensor[B, Q, 8]      encoded box predictions
              'confidence':    Tensor[B, Q, 1]      objectness score
              'query_embeds':  Tensor[B, Q, D]      per-query embeddings
                                                    (used by tracking head)
              'attn_weights':  Tensor[B, Q, N_total] (last layer attention)
        """
        B = scene.B
        memory = scene.tokens    # [B, N_total, D]

        # Initialise queries by expanding learnable parameters over batch
        queries = self.object_queries.expand(B, -1, -1)  # [B, Q, D]

        # Run through decoder layers
        attn_weights = None
        for layer in self.decoder_layers:
            queries, attn_weights = layer(queries, memory)
            # queries: [B, Q, D]

        queries = self.final_norm(queries)  # [B, Q, D]

        # ---- Prediction heads ----------------------------------------
        class_logits = self.class_head(queries)   # [B, Q, C+1]
        pred_boxes   = self.box_head(queries)      # [B, Q, 8]
        confidence   = self.conf_head(queries)     # [B, Q, 1]
        confidence   = torch.sigmoid(confidence)   # [B, Q, 1]

        return {
            'class_logits':  class_logits,    # [B, Q, C+1]
            'pred_boxes':    pred_boxes,       # [B, Q, 8]  (encoded)
            'confidence':    confidence,       # [B, Q, 1]
            'query_embeds':  queries,          # [B, Q, D]  passed to tracking
            'attn_weights':  attn_weights,     # [B, Q, N_total]
        }

    @staticmethod
    def decode_boxes(pred_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert network output to physical box parameters.

        Input  (pred_boxes): Tensor[..., 8]  (cx, cy, cz, log_dx, log_dy, log_dz, sin_yaw, cos_yaw)
        Output:              Tensor[..., 7]  (cx, cy, cz, dx,     dy,     dz,     yaw)

        Using log for size ensures predicted dimensions are always positive.
        Using sin/cos for yaw avoids the π-periodicity discontinuity.
        """
        center = pred_boxes[..., :3]             # (cx, cy, cz)
        size   = pred_boxes[..., 3:6].exp()      # exp(log_d) → positive dims
        sin_y  = pred_boxes[..., 6:7]
        cos_y  = pred_boxes[..., 7:8]
        yaw    = torch.atan2(sin_y, cos_y)       # [-π, π]
        return torch.cat([center, size, yaw], dim=-1)  # [..., 7]
