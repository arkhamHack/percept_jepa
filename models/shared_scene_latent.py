"""
Shared Scene Latent
====================
Container and abstraction for the fused multimodal scene representation.

Design rationale
----------------
Rather than passing raw tensors between modules, we wrap the scene latent
in a structured container.  This makes it easy to:
  1. Route subsets of tokens to different task heads
  2. Extend with temporal memory tokens (world model state)
  3. Cache scene latents for multi-frame inference
  4. Introspect token semantics in research experiments

Future extensions (marked TODO):
  - Temporal memory bank: store latents from previous frames and append
    as "memory tokens" for long-horizon reasoning.
  - Latent future prediction: train an auxiliary head to predict the
    scene latent at t+1 (V-JEPA-style predictive world model).
  - Occupancy prediction: project scene latent to a 3-D voxel grid.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torch.nn as nn


@dataclass
class SceneLatent:
    """
    Structured container for the shared scene representation.

    Attributes:
        tokens:        Tensor[B, N_total, D]   full fused token sequence
        visual_slice:  slice object for visual token positions
        radar_slice:   slice object for radar token positions
        ego_idx:       int index of the ego-motion token
        B:             batch size
        D:             hidden dim
    """
    tokens:       torch.Tensor
    visual_slice: slice
    radar_slice:  slice
    ego_idx:      int

    @property
    def visual_tokens(self) -> torch.Tensor:
        """Tensor[B, T*P, D]  visual portion of the scene latent."""
        return self.tokens[:, self.visual_slice, :]

    @property
    def radar_tokens(self) -> torch.Tensor:
        """Tensor[B, K, D]  radar portion of the scene latent."""
        return self.tokens[:, self.radar_slice, :]

    @property
    def ego_token(self) -> torch.Tensor:
        """Tensor[B, 1, D]  ego-motion token."""
        return self.tokens[:, self.ego_idx:self.ego_idx + 1, :]

    @property
    def B(self) -> int:
        return self.tokens.shape[0]

    @property
    def N(self) -> int:
        return self.tokens.shape[1]

    @property
    def D(self) -> int:
        return self.tokens.shape[2]

    @property
    def device(self) -> torch.device:
        return self.tokens.device


class SharedSceneLatent(nn.Module):
    """
    Wraps the raw fusion transformer output into a structured SceneLatent.

    Also optionally maintains a temporal memory bank of past scene latents
    for multi-frame context (disabled by default in the base PoC).

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        self.use_memory = cfg.model.scene_latent.use_memory_tokens
        self.memory_size = cfg.model.scene_latent.memory_size

        # Optional: learnable memory tokens (TODO — enable for full world model)
        if self.use_memory:
            self.memory_tokens = nn.Parameter(
                torch.randn(1, self.memory_size, self.hidden_dim)
            )
            nn.init.trunc_normal_(self.memory_tokens, std=0.02)
        else:
            self.memory_tokens = None

        # Per-frame memory buffer (not a nn.Parameter — runtime state)
        self._memory_bank: Optional[torch.Tensor] = None  # [B, M, D]

    def forward(
        self,
        fused_tokens: torch.Tensor,   # [B, T*P + K + 1, D]
        n_visual: int,                # T * P
        n_radar:  int,                # K
    ) -> SceneLatent:
        """
        Build a structured SceneLatent from the raw fusion transformer output.

        Args:
            fused_tokens: Tensor[B, N_total, D]
            n_visual:     number of visual tokens (= T * num_patches)
            n_radar:      number of radar tokens (= K)

        Returns:
            SceneLatent with routing information baked in
        """
        visual_slice = slice(0, n_visual)
        radar_slice  = slice(n_visual, n_visual + n_radar)
        ego_idx      = n_visual + n_radar   # last position

        # If memory tokens are enabled, append them to the token sequence.
        # The detection/prediction heads can optionally attend to memory.
        if self.use_memory and self.memory_tokens is not None:
            B = fused_tokens.shape[0]
            mem = self.memory_tokens.expand(B, -1, -1)  # [B, M, D]
            fused_tokens = torch.cat([fused_tokens, mem], dim=1)
            # NOTE: visual/radar/ego slices still point to the same positions

        return SceneLatent(
            tokens=fused_tokens,
            visual_slice=visual_slice,
            radar_slice=radar_slice,
            ego_idx=ego_idx,
        )

    def update_memory(self, scene: SceneLatent):
        """
        Store the current scene latent as memory for the next frame.

        TODO: Implement proper temporal memory update with:
          - Exponential moving average (EMA)
          - Gated update (LSTM-style)
          - Learned memory write heads
        """
        if self.use_memory:
            # Detach to prevent gradients flowing through time (TBPTT)
            self._memory_bank = scene.tokens.detach()

    def clear_memory(self):
        """Reset temporal memory (call at scene boundaries)."""
        self._memory_bank = None
