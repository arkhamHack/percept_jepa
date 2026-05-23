"""
Positional Encodings
=====================
Various positional encoding schemes used across the architecture:
  1. SinusoidalPositionalEncoding1D   — standard 1-D sinusoidal PE
  2. LearnablePositionalEncoding1D    — learnable 1-D PE
  3. FourierPositionalEncoding3D      — Fourier features for 3-D (x,y,z) coords
  4. PatchPositionalEncoding2D        — 2-D spatial patch PE for ViT
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding1D(nn.Module):
    """
    Classic sinusoidal PE (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Input:  Tensor[B, N, D]
    Output: Tensor[B, N, D]  (x + PE)
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: [max_len, D]
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        N = x.shape[1]
        x = x + self.pe[:, :N, :]
        return self.dropout(x)


class LearnablePositionalEncoding1D(nn.Module):
    """
    Learnable 1-D positional embedding.

    Preferred for architectures where the input sequence length is fixed
    and we want the model to adapt its positional biases during training.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.trunc_normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        N = x.shape[1]
        pos = torch.arange(N, device=x.device).unsqueeze(0)  # [1, N]
        x = x + self.pe(pos)
        return self.dropout(x)


class FourierPositionalEncoding3D(nn.Module):
    """
    Random Fourier Features (RFF) positional encoding for 3-D coordinates.

    Maps (x, y, z) real-world positions to a D-dimensional sinusoidal
    feature vector.  Used to inject spatial position into radar tokens.

    NeRF-style: PE(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^L π p), cos(2^L π p)]

    Args:
        d_model:  output dimension
        max_freq: number of frequency octaves
    """

    def __init__(self, d_model: int, max_freq: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

        # Number of Fourier frequencies per xyz component
        n_freqs = d_model // 6  # 3 dims × 2 (sin+cos)
        if n_freqs == 0:
            n_freqs = 1
        self.n_freqs = n_freqs

        freq_bands = 2.0 ** torch.linspace(0, max_freq, n_freqs)
        self.register_buffer('freq_bands', freq_bands)   # [n_freqs]

        # Linear projection to exact d_model
        raw_dim = 3 * n_freqs * 2  # sin + cos for x, y, z
        self.proj = nn.Linear(raw_dim, d_model)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Tensor[..., 3]   real-world 3-D coordinates (metres)

        Returns:
            Tensor[..., D]   positional encoding
        """
        # Normalise coordinates to [-1, 1] (assuming ±50m scene range)
        xyz_n = xyz / 50.0   # rough normalisation

        feats = []
        for dim in range(3):
            coord = xyz_n[..., dim:dim + 1]   # [..., 1]
            for freq in self.freq_bands:
                feats.append(torch.sin(math.pi * freq * coord))
                feats.append(torch.cos(math.pi * freq * coord))

        feats = torch.cat(feats, dim=-1)   # [..., 3 * n_freqs * 2]
        return self.proj(feats)            # [..., D]


class PatchPositionalEncoding2D(nn.Module):
    """
    2-D learnable positional encoding for ViT patch grids.

    Rows and columns get independent sinusoidal embeddings that are
    concatenated and projected, following the DeiT design.

    For a grid of H_p × W_p patches, assigns unique positional vectors.
    """

    def __init__(self, d_model: int, h_patches: int, w_patches: int):
        super().__init__()
        self.h = h_patches
        self.w = w_patches

        # Row and column embeddings (learned)
        self.row_embed = nn.Embedding(h_patches, d_model // 2)
        self.col_embed = nn.Embedding(w_patches, d_model // 2)
        nn.init.trunc_normal_(self.row_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.col_embed.weight, std=0.02)

        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor[B, H_p*W_p, D]  patch tokens

        Returns:
            Tensor[B, H_p*W_p, D]  tokens + 2-D positional encoding
        """
        H, W = self.h, self.w
        device = x.device

        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)

        row_emb = self.row_embed(rows)   # [H, D/2]
        col_emb = self.col_embed(cols)   # [W, D/2]

        # Broadcast to [H, W, D]
        pe = torch.cat([
            row_emb.unsqueeze(1).expand(H, W, -1),
            col_emb.unsqueeze(0).expand(H, W, -1),
        ], dim=-1)  # [H, W, D]

        pe = pe.view(H * W, -1).unsqueeze(0)  # [1, P, D]
        pe = self.proj(pe)

        return x + pe
