"""Multimodal perception model: V-JEPA + radar BEV fusion.

Primary pipeline (MultimodalPerceptionModel):
    camera → V-JEPA encoder → image features
    radar  → CUDA BEV voxelization → CNN → BEV features
    [image features, BEV features] → fusion → fused features
    fused features → anchor-free detection head (heatmap + box)
    fused features → velocity head
    fused features → tracking embedding head (optional)

Legacy pipeline (JEPAModel) is kept for backward compatibility and
the optional JEPA fine-tuning path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import VJEPAEncoder, BEVRadarEncoder, ImageEncoder, RadarEncoder
from .heads import (
    ConcatFusion,
    GatedFusion,
    AnchorFreeDetectionHead,
    SpatialVelocityHead,
    TrackingEmbeddingHead,
    ObjectDecoder,
    DetectionHead,
    VelocityHead,
    TrajectoryHead,
)


class MultimodalPerceptionModel(nn.Module):
    """Multimodal perception with pretrained V-JEPA + radar BEV fusion.

    Args:
        vjepa_cfg: V-JEPA encoder config dict.
        radar_cfg: Radar BEV encoder config dict.
        fusion_cfg: Fusion module config dict.
        detection_cfg: Detection head config dict.
        velocity_cfg: Velocity head config dict.
        tracking_cfg: Tracking embedding head config dict (optional).
        bev_cfg: BEV grid config dict.
    """

    def __init__(
        self,
        vjepa_cfg: dict | None = None,
        radar_cfg: dict | None = None,
        fusion_cfg: dict | None = None,
        detection_cfg: dict | None = None,
        velocity_cfg: dict | None = None,
        tracking_cfg: dict | None = None,
        bev_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        vc = vjepa_cfg or {}
        rc = radar_cfg or {}
        fc = fusion_cfg or {}
        dc = detection_cfg or {}
        vlc = velocity_cfg or {}
        tc = tracking_cfg or {}
        bc = bev_cfg or {}

        # --- V-JEPA image encoder ---
        self.image_encoder = VJEPAEncoder(
            model_name=vc.get("model_name", "vit_base_patch16_224"),
            pretrained_path=vc.get("pretrained_path"),
            freeze=vc.get("freeze", True),
            unfreeze_last_n=vc.get("unfreeze_last_n", 0),
            feat_dim=vc.get("feat_dim", 768),
        )
        img_channels = vc.get("feat_dim", 768)

        # --- Radar BEV encoder ---
        x_bounds = tuple(bc.get("x_bounds", [-50.0, 50.0]))
        y_bounds = tuple(bc.get("y_bounds", [-50.0, 50.0]))
        self.radar_encoder = BEVRadarEncoder(
            input_dim=rc.get("input_dim", 6),
            bev_channels=rc.get("bev_channels", 64),
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            grid_h=bc.get("grid_h", 200),
            grid_w=bc.get("grid_w", 200),
        )
        bev_channels = rc.get("bev_channels", 64)

        # --- Fusion ---
        fused_dim = fc.get("fused_dim", 256)
        fusion_type = fc.get("type", "concat")
        if fusion_type == "gated":
            self.fusion = GatedFusion(img_channels, bev_channels, fused_dim)
        else:
            self.fusion = ConcatFusion(img_channels, bev_channels, fused_dim)

        # --- Heads ---
        self.det_head = AnchorFreeDetectionHead(
            feat_dim=fused_dim,
            num_classes=dc.get("num_classes", 1),
        )
        self.vel_head = SpatialVelocityHead(feat_dim=fused_dim)

        self.tracking_enabled = tc.get("enabled", False)
        if self.tracking_enabled:
            self.track_head = TrackingEmbeddingHead(
                feat_dim=fused_dim,
                embed_dim=tc.get("embed_dim", 64),
            )
        else:
            self.track_head = None

        # --- Optional JEPA predictor (for future temporal fine-tuning) ---
        self.jepa_predictor: nn.Module | None = None

    def forward(
        self,
        image: torch.Tensor,
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Args:
            image: ``(B, 3, H, W)`` camera image.
            radar_points: ``(B, N, 6)`` radar points.
            radar_mask: ``(B, N)`` valid-point mask.

        Returns:
            Dict with ``heatmap``, ``box_reg``, ``velocity``,
            and optionally ``tracking_embed``.
        """
        # Image features
        f_img = self.image_encoder(image)  # (B, feat_dim, H', W')

        # Radar BEV features
        f_bev = self.radar_encoder(radar_points, radar_mask)  # (B, bev_ch, H'', W'')

        # Fuse
        f_fused = self.fusion(f_img, f_bev)  # (B, fused_dim, H''', W''')

        # Heads
        det_out = self.det_head(f_fused)
        vel_out = self.vel_head(f_fused)

        result: dict[str, torch.Tensor | None] = {
            "heatmap": det_out["heatmap"],
            "box_reg": det_out["box_reg"],
            "velocity": vel_out,
            "tracking_embed": None,
        }

        if self.tracking_enabled and self.track_head is not None:
            result["tracking_embed"] = self.track_head(f_fused)

        return result

    def set_training_stage(self, stage: int, unfreeze_last_n: int = 4) -> None:
        """Switch training stage for the V-JEPA backbone."""
        self.image_encoder.set_stage(stage, unfreeze_last_n)

    def get_backbone_params(self) -> list[nn.Parameter]:
        """Return V-JEPA parameters (for separate LR in stage 2)."""
        return list(self.image_encoder.parameters())

    def get_non_backbone_params(self) -> list[nn.Parameter]:
        """Return all non-backbone parameters."""
        backbone_ids = {id(p) for p in self.image_encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
#  Legacy JEPAModel (kept for backward compatibility / baselines)
# ──────────────────────────────────────────────────────────────

class JEPAModel(nn.Module):
    """Legacy JEPA model with temporal prediction (ResNet18 backbone).

    Kept for camera-only baseline and optional JEPA self-supervised path.
    """

    def __init__(
        self,
        pretrained: bool = True,
        light: bool = False,
        image_feat_dim: int = 512,
        radar_dim: int = 128,
        latent_dim: int = 256,
        max_objects: int = 50,
        num_modes: int = 5,
        pred_steps: int = 12,
        agent_state_dim: int = 3,
        decoder_heads: int = 8,
        decoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.context_image_encoder = ImageEncoder(pretrained=pretrained, light=light, out_channels=image_feat_dim)
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.radar_encoder = RadarEncoder(radar_dim=radar_dim)

        self.context_proj = nn.Sequential(
            nn.Linear(image_feat_dim + radar_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.context_proj[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.context_proj[0].bias)

        self.target_image_encoder = ImageEncoder(pretrained=pretrained, light=light, out_channels=image_feat_dim)
        self.target_pool = nn.AdaptiveAvgPool2d(1)
        self.target_proj = nn.Linear(image_feat_dim, latent_dim)

        for p in self.target_image_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        self.object_decoder = ObjectDecoder(
            feat_dim=latent_dim, in_channels=image_feat_dim,
            max_objects=max_objects, num_heads=decoder_heads, num_layers=decoder_layers,
        )
        self.det_head = DetectionHead(feat_dim=latent_dim)
        self.vel_head = VelocityHead(feat_dim=latent_dim)
        self.traj_head = TrajectoryHead(
            feat_dim=latent_dim, agent_state_dim=agent_state_dim,
            num_modes=num_modes, pred_steps=pred_steps,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996) -> None:
        for tp, cp in zip(self.target_image_encoder.parameters(), self.context_image_encoder.parameters()):
            tp.data.mul_(momentum).add_(cp.data, alpha=1.0 - momentum)
        ctx_w = self.context_proj[0].weight[:self.latent_dim, :self.target_proj.in_features]
        ctx_b = self.context_proj[0].bias[:self.latent_dim]
        self.target_proj.weight.data.mul_(momentum).add_(ctx_w.data, alpha=1.0 - momentum)
        self.target_proj.bias.data.mul_(momentum).add_(ctx_b.data, alpha=1.0 - momentum)

    def forward(
        self,
        image: torch.Tensor,
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
        future_image: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        feat_map = self.context_image_encoder(image)
        img_feat = self.context_pool(feat_map).flatten(1)
        radar_feat = self.radar_encoder(radar_points, radar_mask)
        z_t = self.context_proj(torch.cat([img_feat, radar_feat], dim=1))
        z_pred = self.predictor(z_t)

        z_target = None
        if future_image is not None:
            fm = self.target_image_encoder(future_image)
            z_target = self.target_proj(self.target_pool(fm).flatten(1)).detach()

        obj_features = self.object_decoder(feat_map)
        B, M = z_t.shape[0], obj_features.shape[1]
        if agent_states is None:
            agent_states = torch.zeros(B, M, 3, device=z_t.device)
        trajectories, traj_logits = self.traj_head(obj_features, agent_states)

        return {
            "z_context": z_t,
            "z_predicted": z_pred,
            "z_target": z_target,
            "boxes": self.det_head(obj_features),
            "velocity": self.vel_head(obj_features),
            "trajectories": trajectories,
            "traj_logits": traj_logits,
        }

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
