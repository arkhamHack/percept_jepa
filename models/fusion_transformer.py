"""
Fusion Transformer
===================
The central module of the architecture.  Fuses visual, radar, and
ego-motion tokens into a unified "shared scene latent" representation.

Design Philosophy
-----------------

Why use a transformer for fusion instead of concatenation or late fusion?
  - Attention allows every token to attend to every other token regardless
    of modality.  This is essential because radar and camera observe
    complementary aspects of the same scene: radar provides precise depth
    and velocity while camera provides rich semantics.
  - In latent space (post V-JEPA encoding), both modalities are already in
    "feature space" rather than raw pixel/point space.  Cross-modal
    attention in latent space is fundamentally different from classical
    sensor fusion — it learns to *align semantics* (e.g. "this visual
    patch looks like a car" ↔ "this radar cluster has car-like velocity").
  - Temporal context is already baked into the visual tokens via V-JEPA
    temporal attention.  The fusion transformer only needs to handle
    spatial and cross-modal alignment.

Why is latent fusion preferred?
  1. Generative flexibility: the fused latent can later be used to
     predict future latents (world model style).
  2. Robustness: if one modality is unavailable, the other still produces
     a usable scene latent.
  3. Representation efficiency: post-encoding, both modalities live in a
     semantically structured space that is more amenable to attention than
     raw features.

How does radar align with visual semantics?
  - There is no strict geometric projection applied here.  Instead,
    we rely on the transformer to learn the correspondence via positional
    encodings that encode real-world 3-D coordinates.
  - Radar tokens carry explicit (x, y, z) position + Doppler velocity.
    Visual tokens carry implicit spatial position from patch indices.
  - By having both modalities attend to each other with positional encoding,
    the model can learn "radar point at (x, y) → attend to visual patch at
    projected (u, v)".  This is softer than geometric projection but more
    adaptable to sensor misalignment and occlusion.

Architecture
------------

Input:
  visual_tokens:  [B, T*P, D]   T=4 frames, P=196 patches per frame
  radar_tokens:   [B, K, D]     K=64 aggregated radar tokens
  ego_motion:     [B, 6]        velocity, acceleration, yaw-rate, speed

Processing:
  1. Ego-motion token: MLP([B,6]) → [B,1,D]
  2. Modality type embeddings: learnable additive embeddings per source
  3. Sequence: [visual_tokens | radar_tokens | ego_token]  [B, T*P+K+1, D]
  4. Optional: cross-modal attention blocks (radar attends visual, visual
     attends radar) before the main self-attention blocks
  5. Full self-attention transformer encoder (L layers)

Output:
  scene_latent:  [B, T*P+K+1, D]  the fused representation

TODO:
  - Deformable attention for efficiency on long visual sequences.
  - Hierarchical fusion (coarse-to-fine temporal scale).
  - Masked modality training for robustness.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


class EgoMotionEncoder(nn.Module):
    """
    Encode scalar ego-vehicle kinematics into a single latent token.

    Input:  [B, E]     E = ego feature dimension (vx,vy,ax,ay,yaw_rate,speed)
    Output: [B, 1, D]  single "ego-motion token"
    """

    def __init__(self, cfg):
        super().__init__()
        ecfg = cfg.model.ego
        self.net = nn.Sequential(
            nn.Linear(ecfg.input_dim, ecfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(ecfg.hidden_dim),
            nn.Linear(ecfg.hidden_dim, ecfg.output_dim),
            nn.LayerNorm(ecfg.output_dim),
        )

    def forward(self, ego: torch.Tensor) -> torch.Tensor:
        # ego: [B, 6]
        tok = self.net(ego)          # [B, D]
        return tok.unsqueeze(1)      # [B, 1, D]


class CrossModalAttentionBlock(nn.Module):
    """
    Single cross-modal attention block.

    Query comes from one modality, key/value from another.
    Used to explicitly let radar tokens query visual context (and vice versa)
    before the full self-attention pass.

    Output: updated query tokens, same shape as input.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,    # [B, Nq, D]  — modality being updated
        context: torch.Tensor,  # [B, Nk, D]  — modality providing context
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, Nk]
    ) -> torch.Tensor:
        # Cross-attention: query reads from context
        residual = query
        q = self.norm1(query)
        updated, _ = self.cross_attn(
            q, context, context,
            key_padding_mask=key_padding_mask,
        )
        query = residual + self.drop(updated)

        # FFN
        residual = query
        query = residual + self.drop(self.ff(self.norm2(query)))
        return query


class FusionTransformerLayer(nn.Module):
    """
    A single fusion transformer layer with:
      1. Pre-norm self-attention over the full concatenated token sequence
      2. Pre-norm feed-forward network

    Pre-norm (norm before attention) is used as it is more stable for
    training deep transformers (Xiong et al. 2020).
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, N_total, D]
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.drop(x)

        residual = x
        x = self.norm2(x)
        x = residual + self.drop(self.ff(x))
        return x


class FusionTransformer(nn.Module):
    """
    Multimodal Fusion Transformer.

    Produces a unified "shared scene latent" by jointly attending over
    visual, radar, and ego-motion tokens.

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        fcfg = cfg.model.fusion
        D      = cfg.model.hidden_dim
        L      = fcfg.num_layers
        H      = fcfg.num_heads
        ffn    = fcfg.ffn_dim
        drop   = fcfg.dropout
        cross  = fcfg.use_cross_modal_attention

        self.hidden_dim = D

        # ---- Ego-motion encoder (wraps scalar kinematics → token) ----
        self.ego_encoder = EgoMotionEncoder(cfg)

        # ---- Learnable modality-type embeddings ----------------------
        # Additive embeddings that inform the model which modality each
        # token belongs to (visual / radar / ego).
        self.modality_embed = nn.Embedding(3, D)
        # 0 = visual, 1 = radar, 2 = ego

        # ---- Optional cross-modal attention (radar ↔ visual) --------
        self.use_cross_modal = cross
        if cross:
            self.radar_on_visual = CrossModalAttentionBlock(D, H, ffn, drop)
            self.visual_on_radar = CrossModalAttentionBlock(D, H, ffn, drop)

        # ---- Main self-attention transformer encoder -----------------
        self.layers = nn.ModuleList([
            FusionTransformerLayer(D, H, ffn, drop)
            for _ in range(L)
        ])
        self.final_norm = nn.LayerNorm(D)

    # ------------------------------------------------------------------

    def _add_modality_embed(
        self,
        visual: torch.Tensor,   # [B, Nv, D]
        radar: torch.Tensor,    # [B, Nr, D]
        ego: torch.Tensor,      # [B, 1,  D]
    ):
        """Add learnable modality-type embeddings to distinguish token sources."""
        vis_emb  = self.modality_embed(torch.zeros(1, dtype=torch.long, device=visual.device))
        rad_emb  = self.modality_embed(torch.ones(1,  dtype=torch.long, device=radar.device))
        ego_emb  = self.modality_embed(2 * torch.ones(1, dtype=torch.long, device=ego.device))

        visual = visual + vis_emb   # [B, Nv, D]
        radar  = radar  + rad_emb   # [B, Nr, D]
        ego    = ego    + ego_emb   # [B, 1,  D]
        return visual, radar, ego

    def forward(
        self,
        visual_tokens: torch.Tensor,    # [B, T*P, D]
        radar_tokens:  torch.Tensor,    # [B, K,   D]
        ego_motion:    torch.Tensor,    # [B, 6]
        radar_mask:    Optional[torch.Tensor] = None,  # [B, N] radar valid mask
    ) -> torch.Tensor:
        """
        Fuse all modality tokens into a shared scene latent.

        Args:
            visual_tokens: Tensor[B, T*P, D]   spatio-temporal visual tokens
            radar_tokens:  Tensor[B, K, D]      radar latent tokens
            ego_motion:    Tensor[B, 6]          ego kinematics
            radar_mask:    Tensor[B, K]          (optional) valid radar token mask
                           (typically all ones since radar encoder produces fixed K)

        Returns:
            Tensor[B, T*P + K + 1, D]  shared scene latent
              — concatenation of updated visual, radar, and ego tokens

        Token layout in output:
          positions  0 … T*P-1       : visual tokens
          positions  T*P … T*P+K-1   : radar tokens
          position   T*P+K           : ego-motion token
        """
        B = visual_tokens.shape[0]

        # ---- Ego-motion token ----------------------------------------
        ego_token = self.ego_encoder(ego_motion)  # [B, 1, D]

        # ---- Modality type embeddings --------------------------------
        visual_tokens, radar_tokens, ego_token = self._add_modality_embed(
            visual_tokens, radar_tokens, ego_token
        )

        # ---- Cross-modal attention (optional) -----------------------
        # Radar tokens query visual context: lets radar learn to associate
        # with semantically matching visual patches before full fusion.
        if self.use_cross_modal:
            radar_tokens  = self.radar_on_visual(radar_tokens,  visual_tokens)
            visual_tokens = self.visual_on_radar(visual_tokens, radar_tokens)

        # ---- Concatenate all tokens ----------------------------------
        # Layout: [visual | radar | ego]
        tokens = torch.cat([visual_tokens, radar_tokens, ego_token], dim=1)
        # tokens: [B, T*P + K + 1, D]

        Nv = visual_tokens.shape[1]   # T * P
        Nr = radar_tokens.shape[1]    # K
        N_total = tokens.shape[1]     # T*P + K + 1

        # ---- Key padding mask (mask out no real tokens here) ---------
        # All visual and ego tokens are always valid.
        # Radar tokens could be masked if K comes from a variable source,
        # but our encoder produces K valid tokens always, so mask is None.
        # Extend this if you want to mask out empty radar scans.
        key_padding_mask = None

        # ---- Self-attention fusion transformer -----------------------
        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask)
            # tokens: [B, N_total, D]

        tokens = self.final_norm(tokens)
        # tokens: [B, T*P + K + 1, D]

        return tokens  # shared scene latent
