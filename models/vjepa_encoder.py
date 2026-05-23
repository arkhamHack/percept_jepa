"""
V-JEPA Encoder Wrapper
=======================
Wraps a pretrained Vision Transformer backbone (V-JEPA or timm ViT) to
produce dense spatio-temporal latent tokens suitable for downstream fusion.

Architecture overview
---------------------

Input: [B, T, C, H, W]  — batch of T-frame video clips

Step 1 — Spatial Patch Embedding (per frame)
  Each frame [B*T, C, H, W] is processed by a standard ViT patch embedding.
  For patch_size=16, img_size=224:
    num_patches (per frame) = (224/16)^2 = 196

  Per-frame spatial tokens: [B*T, num_patches, D]

Step 2 — (Optional) V-JEPA Spatial Attention Blocks
  If pretrained V-JEPA weights are loaded, they provide spatially rich
  representations encoding semantic structure without explicit supervision
  (prediction in latent space, NOT pixel reconstruction).
  Output: [B*T, num_patches, D]

Step 3 — Temporal Transformer Layers
  Reshape to [B, num_patches, T, D], then apply self-attention along the
  time axis for each patch position independently (factored temporal attn).
  This captures temporal motion cues while keeping memory manageable.
  Output: [B, T*num_patches, D]  (flattened back)

Key design choices
------------------
• We keep V-JEPA as a *context encoder* (not the predictor).
  The predictor branch is not needed; we only want rich dense tokens.
• Frozen backbone: during early training we only train the temporal layers
  and downstream heads. Fine-tuning is enabled after `unfreeze_epoch`.
• Dense tokens (NOT CLS-pooled): downstream fusion and detection need
  spatially localised features.

TODO:
  - Add masked patch prediction loss for self-supervised fine-tuning.
  - Support V-JEPA-H (ViT-H/14) larger backbone.
  - Hierarchical temporal attention across multi-scale feature maps.
"""

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available — V-JEPA encoder will use minimal ViT stub.")

try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModel = None
    TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TemporalTransformerLayer(nn.Module):
    """
    Single transformer layer operating along the time axis.

    Input shape:  [B * num_patches, T, D]
    Output shape: [B * num_patches, T, D]

    By factoring temporal attention per patch position we keep O(T^2)
    complexity rather than O((T*P)^2) global space-time attention.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BP, T, D]  (B*num_patches, T, D)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + self.drop(x)

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + self.drop(x)
        return x


class TemporalPositionalEncoding(nn.Module):
    """
    Learnable 1-D positional embedding added along the temporal dimension.

    Shape: [1, max_len, D]  (broadcast over batch).
    """

    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BP, T, D]
        T = x.shape[1]
        return x + self.pe[:, :T, :]


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class VJEPAEncoder(nn.Module):
    """
    V-JEPA-inspired temporal visual encoder.

    Outputs dense spatio-temporal token sequences for downstream fusion.

    Args:
        cfg: model config (cfg.model.vjepa sub-section expected)
    """

    def __init__(self, cfg):
        super().__init__()
        vcfg = cfg.model.vjepa
        self.hidden_dim  = cfg.model.hidden_dim
        self.backbone_name = vcfg.backbone
        self.backend = getattr(vcfg, 'backend', 'timm_fallback')
        self.patch_size  = vcfg.patch_size
        self.img_size    = vcfg.img_size
        self.frozen      = vcfg.frozen
        self.official_model_id = getattr(vcfg, 'official_model_id', '')
        self.official_local_path = getattr(vcfg, 'official_local_path', '')
        self.official_trust_remote_code = bool(getattr(vcfg, 'official_trust_remote_code', True))
        self.official_outputs_temporal_tokens = bool(
            getattr(vcfg, 'official_outputs_temporal_tokens', True)
        )
        self.apply_temporal_on_official = bool(
            getattr(vcfg, 'apply_temporal_on_official', False)
        )

        # ---- Spatial backbone ----------------------------------------
        # Backend modes:
        #   - timm_fallback: current PoC-compatible ViT path
        #   - official_vjepa2: load official frozen encoder via HF/local path
        self.spatial_encoder = None
        self.official_encoder = None
        if self.backend == 'official_vjepa2':
            self.official_encoder, self.spatial_dim = self._build_official_vjepa2_encoder()
            logger.info("Using backend=official_vjepa2")
        else:
            self.spatial_encoder, self.spatial_dim = self._build_spatial_encoder(
                vcfg.backbone, pretrained=vcfg.pretrained
            )
            logger.info("Using backend=timm_fallback")

        # Number of spatial tokens per frame (excluding CLS)
        num_h = self.img_size // self.patch_size
        self.num_patches = num_h * num_h   # e.g. 14*14=196 for ViT-B/16 with img_size=224

        # ---- Projection to hidden_dim (if backbone dim != hidden_dim) --
        if self.spatial_dim != self.hidden_dim:
            self.spatial_proj = nn.Linear(self.spatial_dim, self.hidden_dim)
        else:
            self.spatial_proj = nn.Identity()

        # ---- Temporal transformer layers -----------------------------
        # These are *always* trained; the spatial backbone may be frozen.
        T_depth  = vcfg.temporal_depth
        T_heads  = vcfg.temporal_heads
        T_drop   = vcfg.temporal_dropout
        ffn_dim  = self.hidden_dim * 4

        self.temporal_pe = TemporalPositionalEncoding(self.hidden_dim, max_len=32)
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerLayer(self.hidden_dim, T_heads, ffn_dim, T_drop)
            for _ in range(T_depth)
        ])
        self.temporal_norm = nn.LayerNorm(self.hidden_dim)

        # ---- Optionally freeze spatial backbone ----------------------
        if self.frozen:
            self._freeze_backbone()

    # ------------------------------------------------------------------

    def _build_spatial_encoder(
        self, model_name: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """
        Build the spatial ViT encoder.

        Priority:
          1. Attempt to load official V-JEPA checkpoint (expects
             VJEPA_WEIGHTS env var to point to the .pt file).
          2. Fall back to timm ImageNet-pretrained ViT.

        Returns:
            (encoder_module, output_feature_dim)
        """
        import os

        vjepa_ckpt = os.environ.get('VJEPA_WEIGHTS', '')
        if vjepa_ckpt and os.path.isfile(vjepa_ckpt):
            logger.info(f"Loading V-JEPA weights from {vjepa_ckpt}")
            return self._load_vjepa_checkpoint(vjepa_ckpt)

        if TIMM_AVAILABLE:
            logger.info(
                f"V-JEPA weights not found. Using timm '{model_name}' "
                f"(pretrained={pretrained}) as backbone."
            )
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,         # remove classification head
                global_pool='',        # disable global pooling → get patch tokens
            )
            # timm ViT embed_dim
            embed_dim = model.embed_dim
            return model, embed_dim

        # Minimal stub (no timm)
        logger.warning("Using minimal ViT stub (no pretrained weights).")
        stub = _MinimalViT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.hidden_dim,
        )
        return stub, self.hidden_dim

    def _build_official_vjepa2_encoder(self) -> Tuple[nn.Module, int]:
        """
        Build official V-JEPA2/2.1 encoder backend.

        Loading priority:
          1. model.vjepa.official_local_path
          2. model.vjepa.official_model_id (HF)

        The loaded model is used as a frozen feature extractor.
        """
        source = self.official_local_path or self.official_model_id
        if not source:
            raise ValueError(
                "backend=official_vjepa2 requires either model.vjepa.official_local_path "
                "or model.vjepa.official_model_id"
            )

        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is required for backend=official_vjepa2. "
                "Install with: pip install transformers"
            )

        logger.info(f"Loading official V-JEPA model from: {source}")
        model = AutoModel.from_pretrained(
            source,
            trust_remote_code=self.official_trust_remote_code,
        )

        # Try common hidden-dim fields across HF model configs.
        cfg = getattr(model, 'config', None)
        embed_dim = None
        for key in ('hidden_size', 'embed_dim', 'encoder_embed_dim', 'd_model'):
            if cfg is not None and hasattr(cfg, key):
                embed_dim = int(getattr(cfg, key))
                break
        if embed_dim is None:
            # Conservative fallback for ViT-B style dimensions.
            embed_dim = self.hidden_dim
            logger.warning(
                "Could not infer official encoder embed_dim from config; "
                f"defaulting to hidden_dim={self.hidden_dim}."
            )

        return model, embed_dim

    def _load_vjepa_checkpoint(self, path: str) -> Tuple[nn.Module, int]:
        """
        Load an official V-JEPA encoder checkpoint.

        The official repo saves the encoder as a timm ViT dict under the
        'encoder' key.  Adjust this if your checkpoint format differs.
        """
        ckpt = torch.load(path, map_location='cpu')
        # Try standard V-JEPA checkpoint format
        state_dict = ckpt.get('encoder', ckpt)

        if TIMM_AVAILABLE:
            model = timm.create_model(
                self.backbone_name, pretrained=False,
                num_classes=0, global_pool=''
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys: {missing[:5]}")
            return model, model.embed_dim

        raise RuntimeError(
            "timm is required to load V-JEPA checkpoints. "
            "Install with: pip install timm"
        )

    def _freeze_backbone(self):
        """Freeze all spatial encoder parameters."""
        backbone = self.official_encoder if self.backend == 'official_vjepa2' else self.spatial_encoder
        for p in backbone.parameters():
            p.requires_grad_(False)
        logger.info("V-JEPA spatial backbone is FROZEN.")

    def unfreeze_backbone(self):
        """Unfreeze the spatial encoder (called after warmup)."""
        backbone = self.official_encoder if self.backend == 'official_vjepa2' else self.spatial_encoder
        for p in backbone.parameters():
            p.requires_grad_(True)
        logger.info("V-JEPA spatial backbone UNFROZEN for fine-tuning.")

    def backbone_parameters(self):
        """Return active visual-backbone parameters for optimiser grouping."""
        backbone = self.official_encoder if self.backend == 'official_vjepa2' else self.spatial_encoder
        return backbone.parameters()

    # ------------------------------------------------------------------

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame spatial patch tokens.

        Args:
            x: Tensor[B*T, C, H, W]

        Returns:
            Tensor[B*T, num_patches, D_hidden]
        """
        if TIMM_AVAILABLE and hasattr(self.spatial_encoder, 'forward_features'):
            # timm ViT: forward_features returns [B, 1+num_patches, D] (with CLS)
            features = self.spatial_encoder.forward_features(x)
            # Shape: [B*T, 1 + num_patches, embed_dim]
            # Drop the CLS token at position 0 to keep dense spatial tokens
            if features.dim() == 3 and features.shape[1] > self.num_patches:
                features = features[:, 1:, :]   # [B*T, num_patches, embed_dim]
            elif features.dim() == 3:
                features = features             # already no CLS
        else:
            features = self.spatial_encoder(x)  # fallback stub

        # Project to shared hidden_dim if needed
        features = self.spatial_proj(features)  # [B*T, P, D]
        return features

    def _extract_tokens_from_official_output(self, out: Any) -> torch.Tensor:
        """
        Normalize official model outputs to token tensor.

        Expected outputs (any one of):
          - Tensor[B, N, D]
          - object with .last_hidden_state -> Tensor[B, N, D]
          - dict containing one of keys: last_hidden_state, hidden_states,
            encoder_last_hidden_state, x
        """
        if isinstance(out, torch.Tensor):
            return out

        if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
            return out.last_hidden_state

        if isinstance(out, dict):
            for key in ('last_hidden_state', 'encoder_last_hidden_state', 'x'):
                if key in out and isinstance(out[key], torch.Tensor):
                    return out[key]
            if 'hidden_states' in out and out['hidden_states']:
                return out['hidden_states'][-1]

        raise RuntimeError(
            "Unable to extract token tensor from official V-JEPA2 output. "
            "Expected tensor or an output containing last_hidden_state."
        )

    def _encode_official_vjepa2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode clip with official V-JEPA2/2.1 encoder.

        Args:
            x: Tensor[B, T, C, H, W]

        Returns:
            Tensor[B, T*P, D] token sequence for fusion.
        """
        B, T, C, H, W = x.shape

        # Prefer native video input when supported.
        try:
            out = self.official_encoder(pixel_values=x)
            toks = self._extract_tokens_from_official_output(out)
        except Exception:
            # Fallback: framewise encode then restore temporal ordering.
            x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
            out = self.official_encoder(pixel_values=x_flat)
            toks = self._extract_tokens_from_official_output(out)
            if toks.dim() == 3 and toks.shape[0] == B * T:
                toks = rearrange(toks, '(b t) p d -> b t p d', b=B, t=T)

        # Normalize token layouts.
        # Common cases:
        #   [B, N, D]          -> already flat tokens
        #   [B, T, P, D]       -> flatten T and P
        #   [B*T, P, D]        -> reshape then flatten
        if toks.dim() == 4:
            toks = rearrange(toks, 'b t p d -> b (t p) d')
        elif toks.dim() == 3 and toks.shape[0] == B * T:
            toks = rearrange(toks, '(b t) p d -> b (t p) d', b=B, t=T)
        elif toks.dim() != 3:
            raise RuntimeError(
                f"Unexpected official token shape: {tuple(toks.shape)}"
            )

        # Drop CLS if present (heuristic: token count not divisible by T but N-1 is).
        n_tok = toks.shape[1]
        if n_tok > 1 and (n_tok % T != 0) and ((n_tok - 1) % T == 0):
            toks = toks[:, 1:, :]

        toks = self.spatial_proj(toks)

        # If official encoder already models temporal structure, we typically skip
        # extra temporal blocks. Enable override via config for ablations.
        if self.official_outputs_temporal_tokens and not self.apply_temporal_on_official:
            return toks

        # Optional extra temporal refinement with existing PoC temporal blocks.
        if toks.shape[1] % T != 0:
            raise RuntimeError(
                "Cannot apply PoC temporal blocks on official tokens because token count "
                f"N={toks.shape[1]} is not divisible by T={T}."
            )

        P = toks.shape[1] // T
        toks = rearrange(toks, 'b (t p) d -> (b p) t d', t=T, p=P)
        toks = self.temporal_pe(toks)
        for layer in self.temporal_layers:
            toks = layer(toks)
        toks = self.temporal_norm(toks)
        toks = rearrange(toks, '(b p) t d -> b (t p) d', b=B, p=P)
        return toks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full V-JEPA encoding: spatial patch embedding + temporal attention.

        Args:
            x: Tensor[B, T, C, H, W]
                B = batch size
                T = temporal sequence length (e.g. 4)
                C = 3 (RGB)
                H, W = 224, 224

        Returns:
            Tensor[B, T * num_patches, D_hidden]
                T * num_patches = 4 * 196 = 784  (ViT-B/16, T=4)
                D_hidden = 768

        Explanation of output semantics:
          - Each token corresponds to a 16×16 spatial patch at a particular
            time step.  The temporal transformer has infused cross-frame
            motion context into each token.
          - The tokens are DENSE (no CLS, no global pooling) to preserve
            spatial localisation needed by the detection head.
          - The latent semantics represent V-JEPA's learned representation
            space, which was trained by predicting masked patch embeddings —
            a world-model-style objective that encourages temporally
            consistent, motion-aware features.
        """
        B, T, C, H, W = x.shape

        if self.backend == 'official_vjepa2':
            return self._encode_official_vjepa2(x)

        # ---- Step 1: Spatial encoding per frame ----------------------
        # Merge batch and time dims for efficient parallel processing
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        # x_flat: [(B*T), C, H, W]

        spatial_tokens = self.encode_spatial(x_flat)
        # spatial_tokens: [(B*T), P, D]   P = num_patches

        P = spatial_tokens.shape[1]  # num_patches
        D = spatial_tokens.shape[2]  # hidden_dim

        # ---- Step 2: Temporal attention per patch position -----------
        # Reshape to (B*P, T, D) so each patch position attends across T
        spatial_tokens = rearrange(
            spatial_tokens, '(b t) p d -> (b p) t d', b=B, t=T
        )
        # spatial_tokens: [(B*P), T, D]

        # Add learnable temporal positional embedding
        spatial_tokens = self.temporal_pe(spatial_tokens)
        # spatial_tokens: [(B*P), T, D]

        # Apply temporal transformer layers
        for layer in self.temporal_layers:
            spatial_tokens = layer(spatial_tokens)  # [(B*P), T, D]

        spatial_tokens = self.temporal_norm(spatial_tokens)

        # ---- Step 3: Reshape to flat token sequence ------------------
        # Reshape back: (B*P, T, D) → (B, T*P, D)
        temporal_tokens = rearrange(
            spatial_tokens, '(b p) t d -> b (t p) d', b=B, p=P
        )
        # temporal_tokens: [B, T*P, D]
        # e.g. [B, 784, 768] for T=4, P=196, D=768

        return temporal_tokens


# ---------------------------------------------------------------------------
# Minimal ViT stub (used when timm is not installed)
# ---------------------------------------------------------------------------

class _MinimalViT(nn.Module):
    """
    Bare-bones patch-embedding ViT for testing without timm.
    NOT intended for production or pretrained loading.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        # 4 lightweight attention blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.patch_embed(x)               # [B, D, H/p, W/p]
        B, D, Hh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, P, D]
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.norm(x)
        return x                               # [B, P, D]
