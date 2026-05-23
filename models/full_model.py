"""
Full Model — RearFusionModel
=============================
Assembles all sub-modules into a single PyTorch module that can be trained
end-to-end on the multitask detection + tracking + prediction objective.

Forward pass data flow:

  images [B,T,C,H,W]                    radar_pts [B,N,6]
        ↓                                       ↓
  VJEPAEncoder                          RadarEncoder
        ↓                                       ↓
  visual_tokens [B,T*P,D]           radar_tokens [B,K,D]
        ↓                                       ↓
        └──────────── FusionTransformer ────────────┘
                              ↑
                     ego_token [B,1,D]
                              ↓
                     SharedSceneLatent [B, T*P+K+1, D]
                              ↓
                  ┌──────────┬──────────┐
                  ↓          ↓          ↓
           Detection    Tracking   Prediction
            Head         Head        Head
                  ↓          ↓          ↓
           class_logits  track_emb   traj [B,Q,K,2]
           pred_boxes    [B,Q,E]
           [B,Q,7]
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.vjepa_encoder       import VJEPAEncoder
from models.radar_encoder       import RadarEncoder
from models.fusion_transformer  import FusionTransformer
from models.shared_scene_latent import SharedSceneLatent
from models.detection_head      import DetectionHead
from models.tracking_head       import TrackingHead
from models.prediction_head     import PredictionHead

logger = logging.getLogger(__name__)


class RearFusionModel(nn.Module):
    """
    End-to-end model for rear camera + radar fusion.

    Multi-task outputs:
      - 3-D object detection  (class + box + confidence)
      - Object identity embeddings (for tracking)
      - Future trajectory prediction

    Args:
        cfg: full OmegaConf / dict config
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        D = cfg.model.hidden_dim

        # Compute token counts (needed for scene latent routing)
        T = cfg.data.sequence_length
        img_size   = cfg.model.vjepa.img_size
        patch_size = cfg.model.vjepa.patch_size
        P = (img_size // patch_size) ** 2   # spatial patches per frame
        self.n_visual = T * P               # total visual tokens

        self.n_radar = cfg.model.radar.num_radar_tokens

        # ---- Sub-modules ----------------------------------------
        self.vjepa_encoder      = VJEPAEncoder(cfg)
        self.radar_encoder      = RadarEncoder(cfg)
        self.fusion_transformer = FusionTransformer(cfg)
        self.scene_latent_mod   = SharedSceneLatent(cfg)
        self.detection_head     = DetectionHead(cfg)
        self.tracking_head      = TrackingHead(cfg)
        self.prediction_head    = PredictionHead(cfg)

        logger.info(
            f"RearFusionModel initialised\n"
            f"  Visual tokens per sample:  {self.n_visual}  (T={T}, P={P})\n"
            f"  Radar tokens per sample:   {self.n_radar}\n"
            f"  Total scene latent tokens: {self.n_visual + self.n_radar + 1}\n"
            f"  Hidden dim: {D}"
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        images:      torch.Tensor,               # [B, T, C, H, W]
        radar_pts:   torch.Tensor,               # [B, N, 6]
        radar_mask:  torch.Tensor,               # [B, N]
        ego_motion:  torch.Tensor,               # [B, 6]
    ) -> Dict[str, object]:
        """
        Full forward pass.

        Returns a dict containing all task outputs and intermediate
        representations for loss computation and visualisation.

        Output keys:
          'class_logits'  : Tensor[B, Q, C+1]
          'pred_boxes'    : Tensor[B, Q, 8]   (encoded; use DetectionHead.decode_boxes)
          'confidence'    : Tensor[B, Q, 1]
          'track_embeds'  : Tensor[B, Q, E]
          'pred_traj'     : Tensor[B, Q, K, 2]
          'scene_latent'  : SceneLatent         (for visualisation / auxiliary losses)
          'attn_weights'  : Tensor[B, Q, N]     (detection cross-attn weights)
        """
        # ---- 1. Visual encoding ---------------------------------------
        # images: [B, T, C, H, W]
        visual_tokens = self.vjepa_encoder(images)
        # visual_tokens: [B, T*P, D]

        # ---- 2. Radar encoding ----------------------------------------
        # radar_pts: [B, N, 6]
        radar_tokens = self.radar_encoder(radar_pts, radar_mask)
        # radar_tokens: [B, K, D]

        # ---- 3. Multimodal fusion -------------------------------------
        fused = self.fusion_transformer(
            visual_tokens, radar_tokens, ego_motion, radar_mask=None
        )
        # fused: [B, T*P + K + 1, D]

        # ---- 4. Scene latent container --------------------------------
        scene = self.scene_latent_mod(fused, self.n_visual, self.n_radar)
        # scene.tokens: [B, T*P + K + 1, D]

        # ---- 5. Detection head ----------------------------------------
        det_out = self.detection_head(scene)
        # det_out['class_logits']:  [B, Q, C+1]
        # det_out['pred_boxes']:    [B, Q, 8]
        # det_out['confidence']:    [B, Q, 1]
        # det_out['query_embeds']:  [B, Q, D]
        # det_out['attn_weights']:  [B, Q, N]

        # ---- 6. Tracking head ----------------------------------------
        track_embeds = self.tracking_head(det_out['query_embeds'])
        # track_embeds: [B, Q, E]

        # ---- 7. Prediction head --------------------------------------
        pred_traj = self.prediction_head(det_out['query_embeds'])
        # pred_traj: [B, Q, K, 2]

        return {
            'class_logits': det_out['class_logits'],   # [B, Q, C+1]
            'pred_boxes':   det_out['pred_boxes'],      # [B, Q, 8]
            'confidence':   det_out['confidence'],      # [B, Q, 1]
            'track_embeds': track_embeds,               # [B, Q, E]
            'pred_traj':    pred_traj,                  # [B, Q, K, 2]
            'scene_latent': scene,                      # SceneLatent
            'attn_weights': det_out['attn_weights'],    # [B, Q, N]
        }

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def unfreeze_backbone(self):
        """Enable gradient updates for V-JEPA backbone (after warmup)."""
        self.vjepa_encoder.unfreeze_backbone()

    def freeze_backbone(self):
        """Freeze V-JEPA backbone."""
        self.vjepa_encoder._freeze_backbone()

    def get_parameter_groups(self):
        """
        Return parameter groups with different learning rates.

        Usage in optimiser:
            groups = model.get_parameter_groups()
            optim = torch.optim.AdamW(groups)
        """
        backbone_params = list(self.vjepa_encoder.backbone_parameters())
        backbone_ids    = {id(p) for p in backbone_params}

        other_params = [
            p for p in self.parameters()
            if id(p) not in backbone_ids
        ]

        return [
            {'params': backbone_params, 'lr': self.cfg.training.lr * self.cfg.training.backbone_lr_scale},
            {'params': other_params,    'lr': self.cfg.training.lr},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts per sub-module."""
        def _count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            'vjepa_encoder':      _count(self.vjepa_encoder),
            'radar_encoder':      _count(self.radar_encoder),
            'fusion_transformer': _count(self.fusion_transformer),
            'detection_head':     _count(self.detection_head),
            'tracking_head':      _count(self.tracking_head),
            'prediction_head':    _count(self.prediction_head),
            'total':              _count(self),
        }
