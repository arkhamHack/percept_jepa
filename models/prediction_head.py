"""
Prediction Head — Future Trajectory Decoder
============================================
Predicts K future (x, y) positions for each detected object.

Architecture variants:
  1. MLP decoder (default, fast)
     query_embed [D] → Linear → Hidden → ... → [K*2] → reshape [K, 2]

  2. Transformer decoder (optional, richer)
     K learnable future-step queries attend to the object query embed
     and produce one (x, y) position per step.

Design notes:
--------------
• Input is the per-object query embedding (output of detection head),
  NOT the raw scene tokens.  This ensures predictions are conditioned on
  the object's identity, shape, and context.

• Predictions are *relative displacements* from the current box centre.
  At inference, add to the detected box centre to get absolute positions.
  Using relative coords reduces the regression target range.

• We predict K steps at once (non-autoregressive, like DETR in time).
  Autoregressive generation would be more expressive but slower and
  harder to train with teacher forcing.

• Uncertainty: for a Gaussian-mixture extension, predict K modes with
  mixture weights (like MTP / Trajectron++).  Current PoC predicts the
  single most-likely trajectory.

Output shape: [B, Q, K, 2]
  B = batch size
  Q = number of object queries
  K = future time steps (e.g. 12 steps × 0.5s = 6s horizon)
  2 = (Δx, Δy) relative displacement in ego frame

TODO:
  - Multi-modal trajectory prediction (GMM output head).
  - Recurrent state (GRU) for smoother long-horizon trajectories.
  - Social force / interaction modelling between object queries.
  - Occupancy grid prediction as auxiliary output.
"""

import logging

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class MLPPredictionDecoder(nn.Module):
    """
    Simple MLP trajectory decoder.

    Input:  [B, Q, D]   object query embeddings
    Output: [B, Q, K, 2]  future displacements

    The MLP independently decodes each query's trajectory without
    explicit temporal structure.  Despite its simplicity, MLPs are
    competitive for short-horizon (2-3 s) trajectory prediction.
    """

    def __init__(self, d_model: int, hidden_dim: int, future_steps: int, num_layers: int):
        super().__init__()
        K = future_steps

        layers = [nn.Linear(d_model, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        layers.append(nn.Linear(hidden_dim, K * 2))
        self.net = nn.Sequential(*layers)
        self.K = K

    def forward(self, query_embeds: torch.Tensor) -> torch.Tensor:
        # query_embeds: [B, Q, D]
        B, Q, D = query_embeds.shape
        out = self.net(query_embeds)            # [B, Q, K*2]
        out = out.view(B, Q, self.K, 2)        # [B, Q, K, 2]
        return out


class TransformerPredictionDecoder(nn.Module):
    """
    Transformer-based trajectory decoder.

    K learnable temporal query vectors (one per future step) attend to
    the per-object embedding to produce K trajectory waypoints.

    This is more expressive than MLP because:
      - Each future step can selectively read different aspects of the
        object's context.
      - Future steps attend to each other, modelling temporal smoothness.

    Input:  [B, Q, D]   object query embeddings
    Output: [B, Q, K, 2]
    """

    def __init__(self, d_model: int, hidden_dim: int, future_steps: int, num_layers: int):
        super().__init__()
        K = future_steps

        # Learnable temporal queries (one per future step)
        self.temporal_queries = nn.Parameter(torch.randn(1, K, d_model))
        nn.init.trunc_normal_(self.temporal_queries, std=0.02)

        # Transformer decoder: temporal queries cross-attend to object embed
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Output projection per temporal query: D → 2 (Δx, Δy)
        self.output_proj = nn.Linear(d_model, 2)
        self.K = K

    def forward(self, query_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeds: Tensor[B, Q, D]

        Returns:
            Tensor[B, Q, K, 2]
        """
        B, Q, D = query_embeds.shape
        K = self.K

        # Expand temporal queries over batch × object queries
        tq = self.temporal_queries.expand(B * Q, -1, -1)  # [(B*Q), K, D]

        # Memory = object query embedding (one "token" per object)
        mem = query_embeds.view(B * Q, 1, D)               # [(B*Q), 1, D]

        # Decode: each temporal query attends to the object embedding
        out = self.decoder(tq, mem)                         # [(B*Q), K, D]
        out = self.norm(out)
        out = self.output_proj(out)                         # [(B*Q), K, 2]
        out = out.view(B, Q, K, 2)                          # [B, Q, K, 2]
        return out


class PredictionHead(nn.Module):
    """
    Future trajectory prediction head.

    Wraps either the MLP or Transformer decoder based on config.

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        pcfg = cfg.model.prediction
        D = cfg.model.hidden_dim
        K = pcfg.future_steps
        H = pcfg.hidden_dim
        L = pcfg.decoder_layers
        decoder_type = pcfg.decoder_type

        if decoder_type == 'transformer':
            self.decoder = TransformerPredictionDecoder(D, H, K, L)
        else:
            self.decoder = MLPPredictionDecoder(D, H, K, L)

        self.future_steps = K

    def forward(
        self,
        query_embeds: torch.Tensor,   # [B, Q, D]
    ) -> torch.Tensor:
        """
        Predict future trajectories for all object queries.

        Args:
            query_embeds: Tensor[B, Q, D]  per-object embedding from detection head

        Returns:
            Tensor[B, Q, K, 2]
              B = batch
              Q = num object queries
              K = future steps
              2 = (Δx, Δy) displacement relative to current box centre

        At inference, use the top-scoring queries only:
          abs_pos = box_centres[..., :2] + cumsum(traj_delta, dim=-2)
        """
        traj = self.decoder(query_embeds)   # [B, Q, K, 2]
        return traj
