"""Detection, velocity, tracking, and trajectory heads.

Anchor-free heads operate on spatial feature maps (BEV-fused):
  - AnchorFreeDetectionHead: objectness heatmap + box regression
  - SpatialVelocityHead: per-cell (vx, vy) prediction
  - TrackingEmbeddingHead: per-cell embedding vector for tracking

Legacy per-object heads (ObjectDecoder, DetectionHead, VelocityHead,
TrajectoryHead) are kept for the optional JEPA fine-tuning path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
#  Feature Fusion modules
# ──────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """Concatenate image and BEV features, apply Conv-BN-ReLU fusion block.

    Adapts spatial sizes via adaptive average pooling to the smaller of the
    two inputs before concatenation.
    """

    def __init__(self, img_channels: int, bev_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(img_channels + bev_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.fusion:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, f_img: torch.Tensor, f_bev: torch.Tensor) -> torch.Tensor:
        # Match spatial dims to the smaller one
        h = min(f_img.shape[2], f_bev.shape[2])
        w = min(f_img.shape[3], f_bev.shape[3])
        f_img = nn.functional.adaptive_avg_pool2d(f_img, (h, w))
        f_bev = nn.functional.adaptive_avg_pool2d(f_bev, (h, w))
        return self.fusion(torch.cat([f_img, f_bev], dim=1))


class GatedFusion(nn.Module):
    """Gated fusion: learns a per-channel gate to blend image and BEV features."""

    def __init__(self, img_channels: int, bev_channels: int, out_channels: int) -> None:
        super().__init__()
        self.img_proj = nn.Conv2d(img_channels, out_channels, 1, bias=False)
        self.bev_proj = nn.Conv2d(bev_channels, out_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_img: torch.Tensor, f_bev: torch.Tensor) -> torch.Tensor:
        h = min(f_img.shape[2], f_bev.shape[2])
        w = min(f_img.shape[3], f_bev.shape[3])
        f_img = nn.functional.adaptive_avg_pool2d(f_img, (h, w))
        f_bev = nn.functional.adaptive_avg_pool2d(f_bev, (h, w))

        img_p = self.img_proj(f_img)
        bev_p = self.bev_proj(f_bev)
        g = self.gate(torch.cat([img_p, bev_p], dim=1))
        fused = g * img_p + (1 - g) * bev_p
        return self.out_conv(fused)


# ──────────────────────────────────────────────────────────────
#  Anchor-free spatial heads (operate on fused feature maps)
# ──────────────────────────────────────────────────────────────

class AnchorFreeDetectionHead(nn.Module):
    """Anchor-free detection: objectness heatmap + box regression per cell.

    Outputs:
        heatmap: ``(B, num_classes, H', W')`` objectness logits
        box_reg: ``(B, 4, H', W')`` box offsets ``[dx, dy, w, h]``
    """

    def __init__(self, feat_dim: int = 256, num_classes: int = 1) -> None:
        super().__init__()
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, num_classes, 1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 4, 1),
        )
        # Bias heatmap negative so most cells start as "no object"
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "heatmap": self.heatmap_head(feat),
            "box_reg": self.box_head(feat),
        }


class SpatialVelocityHead(nn.Module):
    """Per-cell velocity prediction ``(vx, vy)``.

    Output: ``(B, 2, H', W')``
    """

    def __init__(self, feat_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class TrackingEmbeddingHead(nn.Module):
    """Per-cell embedding vector for temporal association.

    Output: ``(B, embed_dim, H', W')``
    """

    def __init__(self, feat_dim: int = 256, embed_dim: int = 64) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, embed_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.head(feat), dim=1)


# ──────────────────────────────────────────────────────────────
#  Legacy heads (kept for optional JEPA fine-tuning path)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _sine_pos_2d(
    H: int, W: int, d_model: int, device: torch.device,
) -> torch.Tensor:
    assert d_model % 4 == 0
    d_quarter = d_model // 4
    freq = torch.arange(d_quarter, device=device, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (freq / d_quarter))
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    pe_y = y.unsqueeze(1) * freq.unsqueeze(0)
    pe_x = x.unsqueeze(1) * freq.unsqueeze(0)
    pe_y = torch.cat([pe_y.sin(), pe_y.cos()], dim=-1)
    pe_x = torch.cat([pe_x.sin(), pe_x.cos()], dim=-1)
    pos = torch.cat([
        pe_y.unsqueeze(1).expand(-1, W, -1),
        pe_x.unsqueeze(0).expand(H, -1, -1),
    ], dim=-1)
    return pos.reshape(H * W, d_model).unsqueeze(0)


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(queries, kv, kv)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))
        return queries


class ObjectDecoder(nn.Module):
    def __init__(self, feat_dim=256, in_channels=512, max_objects=50, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_objects = max_objects
        self.input_proj = nn.Conv2d(in_channels, feat_dim, 1)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        self.queries = nn.Embedding(max_objects, feat_dim)
        nn.init.normal_(self.queries.weight, std=0.01)
        self.layers = nn.ModuleList([_DecoderLayer(feat_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, feat_map):
        B = feat_map.shape[0]
        proj = self.input_proj(feat_map)
        _, C, H, W = proj.shape
        kv = proj.flatten(2).permute(0, 2, 1)
        pos = _sine_pos_2d(H, W, C, proj.device)
        kv = kv + pos
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, kv)
        return queries

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class DetectionHead(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, 5))
        self.head[-1].bias.data[4] = -2.0

    def forward(self, obj_features):
        raw = self.head(obj_features)
        coords = raw[..., :4].sigmoid()
        conf = raw[..., 4:5]
        return torch.cat([coords, conf], dim=-1)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class VelocityHead(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, 2))

    def forward(self, obj_features):
        return self.head(obj_features)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())


class TrajectoryHead(nn.Module):
    def __init__(self, feat_dim=256, agent_state_dim=3, num_modes=5, pred_steps=12):
        super().__init__()
        self.num_modes = num_modes
        self.pred_steps = pred_steps
        in_dim = feat_dim + agent_state_dim
        self.traj_mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
            nn.Linear(feat_dim, num_modes * pred_steps * 2),
        )
        self.mode_mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, num_modes),
        )

    def forward(self, obj_features, agent_states):
        B, M, _ = obj_features.shape
        combined = torch.cat([obj_features, agent_states], dim=-1)
        traj = self.traj_mlp(combined).view(B, M, self.num_modes, self.pred_steps, 2)
        logits = self.mode_mlp(combined)
        return traj, logits

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())
