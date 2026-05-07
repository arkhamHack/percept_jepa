"""JEPA model with temporal prediction for radar-camera fusion.

Supports four tasks simultaneously:
  - **Detection**: bounding box + confidence from spatial features
  - **Velocity estimation**: per-object (vx, vy) from spatial features
  - **Trajectory prediction**: multi-modal future waypoints per agent
  - **JEPA self-supervision**: latent z_t -> z_{t+1} prediction (EMA target)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import ImageEncoder, RadarEncoder
from .heads import ObjectDecoder, DetectionHead, VelocityHead, TrajectoryHead


class JEPAModel(nn.Module):
    """Joint-Embedding Predictive Architecture for radar-camera fusion.

    * **Context encoder** encodes the current frame (image + radar).
      - Global pooled feature z_t is used for JEPA self-supervised loss.
      - Spatial feature map is decoded by ObjectDecoder for task heads.
    * **Target encoder** (EMA, frozen) encodes a future frame -> z_{t+1}.
    * **Predictor** maps z_t -> predicted z_{t+1}.
    * **ObjectDecoder** cross-attends learned queries to the spatial feature
      map, producing per-object features that feed into task heads.
    * **Task heads** decode detection, velocity, and trajectory outputs
      from per-object features.

    Args:
        pretrained: Use pretrained ResNet18 weights.
        light: Use light (3-block) ResNet variant.
        image_feat_dim: Image encoder output channels.
        radar_dim: Radar encoder output dimension.
        latent_dim: Shared latent / query dimension.
        max_objects: Max predicted objects (number of decoder queries).
        num_modes: Number of predicted trajectory modes.
        pred_steps: Trajectory prediction horizon in timesteps.
        agent_state_dim: Per-agent state vector dimension.
        decoder_heads: Number of attention heads in ObjectDecoder.
        decoder_layers: Number of cross-attention layers in ObjectDecoder.
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

        # --- context encoder (image + radar -> z_t) ---
        self.context_image_encoder = ImageEncoder(
            pretrained=pretrained, light=light, out_channels=image_feat_dim,
        )
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.radar_encoder = RadarEncoder(radar_dim=radar_dim)

        self.context_proj = nn.Sequential(
            nn.Linear(image_feat_dim + radar_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.context_proj[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.context_proj[0].bias)

        # --- target encoder (image only -> z_{t+1}), updated via EMA ---
        self.target_image_encoder = ImageEncoder(
            pretrained=pretrained, light=light, out_channels=image_feat_dim,
        )
        self.target_pool = nn.AdaptiveAvgPool2d(1)
        self.target_proj = nn.Linear(image_feat_dim, latent_dim)
        nn.init.kaiming_normal_(self.target_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.target_proj.bias)

        for p in self.target_image_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

        # --- predictor: z_t -> z_{t+1}_pred ---
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self._init_predictor()

        # --- object decoder: spatial features -> per-object features ---
        self.object_decoder = ObjectDecoder(
            feat_dim=latent_dim,
            in_channels=image_feat_dim,
            max_objects=max_objects,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
        )

        # --- task heads (operate on per-object features) ---
        self.det_head = DetectionHead(feat_dim=latent_dim)
        self.vel_head = VelocityHead(feat_dim=latent_dim)
        self.traj_head = TrajectoryHead(
            feat_dim=latent_dim,
            agent_state_dim=agent_state_dim,
            num_modes=num_modes,
            pred_steps=pred_steps,
        )

    def _init_predictor(self) -> None:
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996) -> None:
        """EMA update: target <- momentum * target + (1 - momentum) * context."""
        for tp, cp in zip(
            self.target_image_encoder.parameters(),
            self.context_image_encoder.parameters(),
        ):
            tp.data.mul_(momentum).add_(cp.data, alpha=1.0 - momentum)

        ctx_w = self.context_proj[0].weight[:self.latent_dim, :self.target_proj.in_features]
        ctx_b = self.context_proj[0].bias[:self.latent_dim]
        self.target_proj.weight.data.mul_(momentum).add_(ctx_w.data, alpha=1.0 - momentum)
        self.target_proj.bias.data.mul_(momentum).add_(ctx_b.data, alpha=1.0 - momentum)

    def _encode_context(
        self,
        image: torch.Tensor,
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode current frame into global latent and spatial feature map.

        Returns:
            z_t: ``(B, latent_dim)`` global context latent (for JEPA loss).
            feat_map: ``(B, image_feat_dim, H', W')`` spatial features
                (for object decoder / task heads).
        """
        feat_map = self.context_image_encoder(image)
        img_feat = self.context_pool(feat_map).flatten(1)
        radar_feat = self.radar_encoder(radar_points, radar_mask)
        z_t = self.context_proj(torch.cat([img_feat, radar_feat], dim=1))
        return z_t, feat_map

    @torch.no_grad()
    def _encode_target(self, future_image: torch.Tensor) -> torch.Tensor:
        feat_map = self.target_image_encoder(future_image)
        img_feat = self.target_pool(feat_map).flatten(1)
        return self.target_proj(img_feat)

    def forward(
        self,
        image: torch.Tensor,
        radar_points: torch.Tensor,
        radar_mask: torch.Tensor,
        future_image: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Args:
            image: ``(B, 3, H, W)`` current camera image.
            radar_points: ``(B, N, 6)`` current radar points.
            radar_mask: ``(B, N)`` valid-point mask.
            future_image: ``(B, 3, H, W)`` next-frame image (optional).
            agent_states: ``(B, max_objects, 3)`` per-agent state vectors
                (velocity, acceleration, heading change rate). If *None*,
                trajectory prediction uses zero states.

        Returns:
            Dict with ``z_context``, ``z_predicted``, ``z_target``,
            ``boxes``, ``velocity``, ``trajectories``, ``traj_logits``.
        """
        z_t, feat_map = self._encode_context(image, radar_points, radar_mask)
        z_pred = self.predictor(z_t)

        z_target: torch.Tensor | None = None
        if future_image is not None:
            z_target = self._encode_target(future_image).detach()

        # Decode per-object features from spatial map via cross-attention
        obj_features = self.object_decoder(feat_map)  # (B, max_objects, latent_dim)

        B = z_t.shape[0]
        M = obj_features.shape[1]
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
