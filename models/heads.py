"""Object decoder and task heads with spatial cross-attention.

Architecture:
    ObjectDecoder: learned queries cross-attend to the encoder's spatial
    feature map → per-object feature vectors (B, max_objects, feat_dim).

    DetectionHead / VelocityHead: lightweight MLPs on per-object features.

    TrajectoryHead: MLP on per-object features + agent state vectors →
    multi-modal future trajectories.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
#  Positional encoding
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _sine_pos_2d(
    H: int, W: int, d_model: int, device: torch.device,
) -> torch.Tensor:
    """2D sinusoidal positional encoding.

    Returns:
        ``(1, H*W, d_model)`` encoding added to flattened spatial features.
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4"
    d_quarter = d_model // 4

    freq = torch.arange(d_quarter, device=device, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (freq / d_quarter))

    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)

    pe_y = y.unsqueeze(1) * freq.unsqueeze(0)        # (H, d_quarter)
    pe_x = x.unsqueeze(1) * freq.unsqueeze(0)        # (W, d_quarter)

    pe_y = torch.cat([pe_y.sin(), pe_y.cos()], dim=-1)  # (H, d_model//2)
    pe_x = torch.cat([pe_x.sin(), pe_x.cos()], dim=-1)  # (W, d_model//2)

    pos = torch.cat([
        pe_y.unsqueeze(1).expand(-1, W, -1),
        pe_x.unsqueeze(0).expand(H, -1, -1),
    ], dim=-1)  # (H, W, d_model)

    return pos.reshape(H * W, d_model).unsqueeze(0)


# ──────────────────────────────────────────────────────────────
#  Decoder layer
# ──────────────────────────────────────────────────────────────

class _DecoderLayer(nn.Module):
    """Single cross-attention + feed-forward layer."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, queries: torch.Tensor, kv: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.cross_attn(queries, kv, kv)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries


# ──────────────────────────────────────────────────────────────
#  Object decoder (shared)
# ──────────────────────────────────────────────────────────────

class ObjectDecoder(nn.Module):
    """Transforms spatial features into per-object features.

    Learned object queries cross-attend to the encoder's spatial feature
    map. Each query specialises on a different scene region / object,
    producing a rich per-object representation that downstream heads
    can decode into boxes, velocities, or trajectories.

    Args:
        feat_dim: Working dimension for queries and attention.
        in_channels: Channel count of the input spatial feature map.
        max_objects: Number of learned object queries.
        num_heads: Attention heads per layer.
        num_layers: Number of cross-attention + FFN layers.
        dropout: Dropout rate in attention and FFN.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        in_channels: int = 512,
        max_objects: int = 50,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.max_objects = max_objects

        self.input_proj = nn.Conv2d(in_channels, feat_dim, 1)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.queries = nn.Embedding(max_objects, feat_dim)
        nn.init.normal_(self.queries.weight, std=0.01)

        self.layers = nn.ModuleList([
            _DecoderLayer(feat_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Decode per-object features from the spatial feature map.

        Args:
            feat_map: ``(B, in_channels, H', W')`` from the image encoder.

        Returns:
            ``(B, max_objects, feat_dim)`` per-object features.
        """
        B = feat_map.shape[0]

        proj = self.input_proj(feat_map)                # (B, feat_dim, H', W')
        _, C, H, W = proj.shape
        kv = proj.flatten(2).permute(0, 2, 1)           # (B, H'*W', feat_dim)

        pos = _sine_pos_2d(H, W, C, proj.device)        # (1, H'*W', feat_dim)
        kv = kv + pos

        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, kv)

        return queries

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
#  Task heads (operate on per-object features from ObjectDecoder)
# ──────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """Predicts bounding boxes with confidence from per-object features.

    Outputs ``(B, M, 5)`` where each row is ``[x1, y1, x2, y2, conf_logit]``.
    Coordinates are in normalised ``[0, 1]`` space (via sigmoid).
    """

    def __init__(self, feat_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 5),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Bias the confidence logit negative so most queries start as "no object"
        self.head[-1].bias.data[4] = -2.0

    def forward(self, obj_features: torch.Tensor) -> torch.Tensor:
        """Args: obj_features ``(B, M, feat_dim)``.  Returns ``(B, M, 5)``."""
        raw = self.head(obj_features)
        coords = raw[..., :4].sigmoid()
        conf = raw[..., 4:5]
        return torch.cat([coords, conf], dim=-1)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class VelocityHead(nn.Module):
    """Predicts per-object velocities ``(vx, vy)`` from per-object features.

    Outputs ``(B, M, 2)``.
    """

    def __init__(self, feat_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, obj_features: torch.Tensor) -> torch.Tensor:
        """Args: obj_features ``(B, M, feat_dim)``.  Returns ``(B, M, 2)``."""
        return self.head(obj_features)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TrajectoryHead(nn.Module):
    """Multi-modal trajectory prediction head (MTP-style).

    For each object, predicts ``num_modes`` possible future trajectories
    and a probability distribution over modes.

    Uses per-object features from the ObjectDecoder (which encode spatial
    scene context via cross-attention) concatenated with per-agent state
    vectors (velocity, acceleration, heading change rate).

    Outputs:
        trajectories: ``(B, M, num_modes, pred_steps, 2)``
        mode_logits:  ``(B, M, num_modes)``
    """

    def __init__(
        self,
        feat_dim: int = 256,
        agent_state_dim: int = 3,
        num_modes: int = 5,
        pred_steps: int = 12,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.pred_steps = pred_steps

        in_dim = feat_dim + agent_state_dim

        self.traj_mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_modes * pred_steps * 2),
        )
        self.mode_mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, num_modes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.traj_mlp, self.mode_mlp):
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obj_features: torch.Tensor,
        agent_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obj_features: ``(B, M, feat_dim)`` per-object features.
            agent_states: ``(B, M, agent_state_dim)`` per-agent state vectors.

        Returns:
            trajectories ``(B, M, num_modes, pred_steps, 2)`` and
            mode_logits ``(B, M, num_modes)``.
        """
        B, M, _ = obj_features.shape
        combined = torch.cat([obj_features, agent_states], dim=-1)

        traj = self.traj_mlp(combined)
        traj = traj.view(B, M, self.num_modes, self.pred_steps, 2)

        logits = self.mode_mlp(combined)

        return traj, logits

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
